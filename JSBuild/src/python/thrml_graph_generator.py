import os
import hashlib
import equinox as eqx
import jax
import jax.numpy as jnp
from thrml.pgm import AbstractNode
from thrml.block_management import Block
from thrml.block_sampling import (
    BlockGibbsSpec,
    FactorSamplingProgram,
    SamplingSchedule,
    sample_states
)
from thrml.conditional_samplers import AbstractConditionalSampler
from thrml.factor import AbstractFactor
from thrml.interaction import InteractionGroup

# -------------------------------------
# utility
# -------------------------------------

def get_sha256_hash(text: str):
    return hashlib.sha256(text.encode()).hexdigest()


# -------------------------------------
# THRML Node
# -------------------------------------

class ThrmlNode(AbstractNode):
    pass


# -------------------------------------
# Simple container object
# -------------------------------------

class THRMLGraph:

    def __init__(self, working_dir, nodes, program):
        self.working_dir = working_dir
        self.nodes = nodes
        self.program = program


# -------------------------------------
# Graph builder (VERY simple example)
# -------------------------------------

def build_simple_graph(text):

    # naive entity extraction example
    # replace with real NER later
    words = set(text.split())

    nodes = {}
    for w in words:
        nodes[w] = ThrmlNode()

    edges = []
    words = list(words)

    # simple co-occurrence edges
    for i in range(len(words)-1):
        edges.append((words[i], words[i+1]))

    return nodes, edges


# -------------------------------------
# Factor creation
# -------------------------------------
class SimpleInteraction(eqx.Module):
    weight: float


class EdgeFactor(AbstractFactor):

    weight: float

    def __init__(self, weight, blocks):
        super().__init__(blocks)
        self.weight = weight

    def to_interaction_groups(self):
        return [
            InteractionGroup(
                interaction=SimpleInteraction(self.weight),
                head_nodes=self.node_groups[0],
                tail_nodes=[self.node_groups[1]],
            )
        ]


# -------------------------------------
# Sampler
# -------------------------------------

class SimpleSampler(AbstractConditionalSampler):

    def sample(self, key, interactions, active_flags, states, sampler_state, output_sd):

        bias = jnp.zeros(output_sd.shape)

        for interaction, state in zip(interactions, states):

            if isinstance(interaction, SimpleInteraction):

                if len(state) > 0:
                    neighbor = jnp.stack(state, -1)
                    bias += jnp.sum(neighbor, axis=-1)

        noise = jax.random.normal(key, output_sd.shape)

        return bias + noise, sampler_state

    def init(self):
        return None


# -------------------------------------
# MAIN plugin entry point
# -------------------------------------

def generate_thrml(textTitle:str):

    input_hash = get_sha256_hash(textTitle)
    working_dir = f"./{input_hash}"

    os.makedirs(working_dir, exist_ok=True)

    with open(f"./{textTitle}.txt") as f:
        text = f.read()

    nodes, edges = build_simple_graph(text)

    node_list = list(nodes.values())

    block = Block(node_list)

    node_shape_dtypes = {
        ThrmlNode: jax.ShapeDtypeStruct((), jnp.float32)
    }

    spec = BlockGibbsSpec(
        free_blocks=[block],
        clamped_blocks=[],
        node_shape_dtypes=node_shape_dtypes
    )

    # build factors
    factors = []

    for src, dst in edges:

        factors.append(
            EdgeFactor(
                weight=1.0,
                blocks=(Block([nodes[src]]), Block([nodes[dst]]))
            )
        )

    sampler = SimpleSampler()

    program = FactorSamplingProgram(
        gibbs_spec=spec,
        samplers=[sampler],
        factors=factors,
        other_interaction_groups=[]
    )

    return THRMLGraph(working_dir, nodes, program)


# -------------------------------------
# Query helper
# -------------------------------------

def run(graph:THRMLGraph, steps=500):
    key      = jax.random.key(0)
    schedule = SamplingSchedule(
        n_warmup=10,
        n_samples=steps,
        steps_per_sample=2
    )

    init_state = [
        jax.random.normal(key, (1, len(graph.nodes)))
    ]

    samples = sample_states(
        key,
        graph.program,
        schedule,
        init_state
    )

    return samples

def run_query(given_pdf_path, DOMAIN, model_name, token):
    import os
    #from PyPDF2 import PdfReader
    from pypdf import PdfReader
    from fast_graphrag import generate_grag
    from thrml_graph_generator import (
        generate_thrml,
        query_probability,
        discover_entity_types,
        discover_relationships,
        discover_queries,
    )
    from langchain_groq import ChatGroq

    # -----------------------------
    # Initialize LLM
    # -----------------------------
    llm = ChatGroq(temperature=0,
        model=model_name #"Llama-3.1-70b-Versatile",
        max_tokens=8000,
        api_key=token #os.environ.get("GROQ_API_KEY")  # Or set manually
    )   

    # -----------------------------
    # Step 1: Load PDF and convert to text
    # -----------------------------
    pdf_path = given_pdf_path #"./WINGS_Investor_Deck.pdf"
    txt_path = pdf_path.replace(".pdf", ".txt")

    if not os.path.exists(txt_path):
        reader = PdfReader(pdf_path)
        text_content = "\n".join(page.extract_text() or "" for page in reader.pages)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text_content)
    else:
        with open(txt_path, "r", encoding="utf-8") as f:
            text_content = f.read()

    # -----------------------------
    # Step 2: Discover entity types, relationships, and example queries
    # -----------------------------
    entity_types    = discover_entity_types(llm, text_content)
    relationships   = discover_relationships(llm, text_content)
    example_queries = discover_queries(llm, text_content)

    print("Discovered Entity Types:", entity_types)
    print("Discovered Relationships:", relationships)
    print("Generated Example Queries:", example_queries)

    # -----------------------------
    # Step 3: Build GraphRAG from the text
    # -----------------------------
    #DOMAIN = "Automatically extracted domain from PDF document"
    grag = generate_grag(
        textTitle=txt_path.replace(".txt",""),
        DOMAIN=DOMAIN,
        ENTITY_TYPES=entity_types,
        EXAMPLE_QUERIES=example_queries
    )

    # Example: test a query against GraphRAG
    response = grag.query("Who is involved in the main project?").response
    print("\nGraphRAG Sample Query Response:\n", response)

    # -----------------------------
    # Step 4: Export GraphRAG graph for THRML
    # -----------------------------
    graph_data  = grag.export_graph()  # Expected format: {"nodes": [...], "edges": [(src,dst), ...]}
    thrml_graph = generate_thrml(graph_data)

    # -----------------------------
    # Step 5: Sample THRML states and query probabilities
    # -----------------------------
    samples    = run(thrml_graph, steps=500)  # Reduce steps for quick demo
    query_node = entity_types[0] if entity_types else "Tim"  # Query first discovered entity or default
    p = query_probability(thrml_graph, samples, query_node)
    print(f"\nP({query_node} positive state):", p)