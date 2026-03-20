from   thrml.block_management import Block
from   thrml.block_sampling import BlockGibbsSpec, FactorSamplingProgram, SamplingSchedule, sample_states
from   thrml.interaction import InteractionGroup
from   thrml.conditional_samplers import AbstractConditionalSampler
from   thrml.factor import AbstractFactor
import jax
import jax.numpy as jnp
import equinox as eqx
from   merkly.mtree import MerkleTree

class GeneNode(eqx.Module):
    """Wraps a gene / hash into a THRML node."""
    value: jnp.ndarray

class GeneFactor(AbstractFactor):
    """Simple edge factor for gene interactions."""
    weight: float
    def __init__(self, weight, blocks):
        super().__init__(blocks)
        self.weight = weight
    def to_interaction_groups(self):
        return [
            InteractionGroup(
                interaction=self.weight,
                head_nodes=self.node_groups[0],
                tail_nodes=[self.node_groups[1]],
            )
        ]

class GeneSampler(AbstractConditionalSampler):
    """Deterministic-ish sampler for genes (predict GA output)."""
    def sample(self, key, interactions, active_flags, states, sampler_state, output_sd):
        # Aggregate neighbors
        bias = jnp.zeros(output_sd.shape)
        for interaction, state in zip(interactions, states):
            bias += interaction * jnp.sum(state, axis=-1)
        # No stochastic noise — deterministic prediction
        return bias, sampler_state
    def init(self):
        return None

def generate_thrml_from_mtree(mtree: MerkleTree):
    """Convert a Merkle Tree of genes into a THRML graph."""
    leaves = mtree.leaves
    nodes  = [GeneNode(value=jnp.array([int.from_bytes(leaf, 'little') % 100])) for leaf in leaves]
    node_index = {i: i for i in range(len(nodes))}
    
    block = Block(nodes)
    spec = BlockGibbsSpec(
        free_blocks=[block],
        clamped_blocks=[],
        node_shape_dtypes={GeneNode: jax.ShapeDtypeStruct((), jnp.float32)}
    )

    factors = []
    for i in range(len(nodes)-1):
        factors.append(GeneFactor(1.0, (Block([nodes[i]]), Block([nodes[i+1]]))))

    program = FactorSamplingProgram(
        gibbs_spec=spec,
        samplers=[GeneSampler()],
        factors=factors,
        other_interaction_groups=[]
    )

    class GraphWrapper:
        nodes      = nodes
        program    = program
        node_index = node_index

        def __init__(self, nodes, program, node_index):
            #super().__init__(node_index)
            self.program = program
            self.nodes   = nodes
            self.node_index = node_index

    return GraphWrapper(nodes, program, node_index)

def predict_ga_from_graph(graph, inputs):
    """Predict what GA would output given THRML-encoded genes."""
    key = jax.random.PRNGKey(0)
    schedule = SamplingSchedule(n_warmup=5, n_samples=1, steps_per_sample=1)
    init_state = [jnp.zeros((1, len(graph.nodes)))]
    samples = sample_states(key, graph.program, schedule, init_state)
    
    gene_values = samples[0][0]  # shape (num_genes,)
    prediction = sum([g * x for g, x in zip(gene_values, inputs)])
    return prediction

# Example GA genes
genes = [4, -2, 3.5, 5, -11, -4.7]
bytes_inputs = [bytes(str(g), 'utf-8') for g in genes]
mtree = MerkleTree(bytes_inputs, lambda x, y: x + y)

def run_ga_optimization(mtree, genes):
    """Run the full demo: Merkle Tree -> THRML graph -> GA prediction."""
    # Convert MerkleTree to THRML graph
    graph = generate_thrml_from_mtree(mtree)

    # Predict GA output deterministically
    predicted_output = predict_ga_from_graph(graph, genes)
    print("Predicted GA output:", predicted_output)