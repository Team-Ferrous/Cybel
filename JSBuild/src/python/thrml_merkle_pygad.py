import os
import jax
import numpy as np
from typing import List, Tuple
from   PIL import Image
from   merkly.mtree import MerkleTree
import jax.numpy as jnp

# THRML imports (from our previous GeneNode/GeneFactor setup)
from python.thrml_graph_generator import EdgeFactor, SimpleSampler, THRMLGraph, ThrmlNode, get_sha256_hash
from   thrml.block_management     import Block
from   thrml.block_sampling       import BlockGibbsSpec, FactorSamplingProgram, SamplingSchedule, sample_states
from   thrml.interaction          import InteractionGroup
from   thrml.conditional_samplers import AbstractConditionalSampler
from   thrml.factor               import AbstractFactor
import equinox as eqx



# THRML Graph generator from Merkle Tree
class GeneNode(eqx.Module):
    value: jnp.ndarray

class GeneFactor(AbstractFactor):
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
    def sample(self, key, interactions, active_flags, states, sampler_state, output_sd):
        bias = jnp.zeros(output_sd.shape)
        for interaction, state in zip(interactions, states):
            bias += interaction * jnp.sum(state, axis=-1)
        return bias, sampler_state
    def init(self):
        return None

class GraphWrapper:
    def __init__(self, nodes, program):
        self.nodes   = nodes
        self.program = program

def build_mtree(inputs: List[int]):
    # Convert inputs to bytes
    bytes_inputs = [bytes(str(x), "utf-8") for x in inputs]
    mhash_function = lambda x, y: x + y
    mtree = MerkleTree(bytes_inputs, mhash_function)
    return mtree

def generate_thrml_from_mtree(working_dir, mtree: MerkleTree):
    leaves = mtree.leaves
    nodes = [GeneNode(value=jnp.array([int.from_bytes(leaf, "little") % 100])) for leaf in leaves]
    block = Block(nodes)
    spec = BlockGibbsSpec(
        free_blocks=[block],
        clamped_blocks=[],
        node_shape_dtypes={GeneNode: jax.ShapeDtypeStruct((), jnp.float32)},
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
    tgraph = THRMLGraph(working_dir, nodes, program)
    return tgraph #GraphWrapper(nodes, program)

def build_merkle_thrml_graph(title:str, inputs: List[int], edges: List[Tuple[str, str]]):
    input_hash = get_sha256_hash(title)
    working_dir = f"./{input_hash}"
    os.makedirs(working_dir, exist_ok=True)

    # Convert inputs to bytes
    mtree = build_mtree(inputs)
    # Create nodes from Merkle leaves
    nodes = [
        ThrmlNode(f"q{i}", leaf)
        for i, leaf in enumerate(mtree.leaves)
    ]

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

    # Create the graph
    tgraph = THRMLGraph(working_dir, nodes, program)

    # Optionally attach Merkle tree to the graph for verification
    tgraph.mtree = mtree
    return tgraph

def build_thrml_graph_from_merkle(title:str, mtree: MerkleTree, edges: List[Tuple[str, str]]):
    input_hash = get_sha256_hash(title)
    working_dir = f"./{input_hash}"
    os.makedirs(working_dir, exist_ok=True)

    # Create nodes from Merkle leaves
    nodes = [
        ThrmlNode(f"q{i}", leaf)
        for i, leaf in enumerate(mtree.leaves)
    ]

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

    # Create the graph
    tgraph = THRMLGraph(working_dir, nodes, program)

    # Optionally attach Merkle tree to the graph for verification
    tgraph.mtree = mtree
    return tgraph

def predict_ga_from_graph(graph, inputs):
    key = jax.random.PRNGKey(0)
    schedule = SamplingSchedule(n_warmup=5, n_samples=1, steps_per_sample=1)
    init_state = [jnp.zeros((1, len(graph.nodes)))]
    samples = sample_states(key, graph.program, schedule, init_state)
    gene_values = samples[0][0]
    prediction = sum([g * x for g, x in zip(gene_values, inputs)])
    return prediction

# --- Deterministic Run ---
def run_image(target_image_path, function_inputs, mtree):
    input_image  = []
    target_image = Image.open(target_image_path).convert("L") #"eg: target.jpg"
    num_genes    = len(function_inputs)

    for iteration in range(1, 2):  # single deterministic iteration
        graph = generate_thrml_from_mtree(target_image_path, mtree)
        solution = [int(n.value[0]) for n in graph.nodes]
        solution_fitness = 1.0 / np.abs(np.sum(np.array(solution) * function_inputs) - 44)
        prediction = predict_ga_from_graph(graph, function_inputs)

        print(f"Iteration {iteration}")
        print("Parameters of the best solution :", solution)
        print("Fitness value of the best solution =", solution_fitness)
        print("Predicted output based on the THRML graph:", prediction)

        im = np.array(target_image)
        fft_mag = np.abs(np.fft.fftshift(np.fft.fft2(im)))

        visual = np.log(fft_mag + 1e-8) * prediction  # log-safety
        visual = (visual - visual.min()) / (visual.max() - visual.min()) * solution[iteration % num_genes]

        input_image = Image.fromarray((visual * 255).astype(np.uint8))
        input_image.save("out.bmp")

        with open("predictions.txt", "w+") as file1:
            file1.write(f"Iteration: {iteration}\n")
            file1.write(f"Best Soln. Parameters: {solution}\n")
            file1.write(f"Best Prediction:       {prediction}\n\n")

        with open("predictions_mtree.txt", "w+") as file1:
            file1.write(f"Solution Index (mtree): {mtree.raw_leaves}\n")

        # Insert to Supabase or other logging system here
        # insert(iteration, solution, prediction)

# --- Deterministic Text Run ---
def run_text(target_text: str, function_inputs: List[int], mtree: MerkleTree):
    """
    Deterministic run on text using THRML graph.
    """

    # Example: convert text to token vector or feature vector
    # This replaces the 'target_image' in your previous workflow
    text_tokens = [ord(c) for c in target_text]  # simplistic char-to-int mapping
    num_genes = len(function_inputs)
    input_hash = get_sha256_hash(target_text)
    working_dir = f"./{input_hash}"
    for iteration in range(1, 2):  # single deterministic iteration
        # Generate THRML graph from mtree
        graph = generate_thrml_from_mtree(working_dir, mtree)
        solution = [int(n.value[0]) for n in graph.nodes]
        
        # Fitness relative to text tokens
        solution_fitness = 1.0 / (1.0 + np.abs(np.sum(np.array(solution) * function_inputs) - sum(text_tokens)))
        
        # Predict output using GA-like prediction
        prediction = predict_ga_from_graph(graph, function_inputs)

        print(f"Iteration {iteration}")
        print("Parameters of the best solution :", solution)
        print("Fitness value of the best solution =", solution_fitness)
        print("Predicted output based on the THRML graph:", prediction)

        # Convert prediction back to text
        # Example: scale prediction to text length
        predicted_length = max(1, int(prediction) % len(target_text))
        predicted_text = ''.join(target_text[i % len(target_text)] for i in range(predicted_length))

        # Save output text
        with open("out.txt", "w+") as f:
            f.write(predicted_text)

        # Logging
        with open("predictions.txt", "w+") as file1:
            file1.write(f"Iteration: {iteration}\n")
            file1.write(f"Best Soln. Parameters: {solution}\n")
            file1.write(f"Best Prediction:       {prediction}\n\n")

        with open("predictions_mtree.txt", "w+") as file1:
            file1.write(f"Solution Index (mtree): {mtree.raw_leaves}\n")

        # Optional: insert into Supabase or other logging system
        # insert(iteration, solution, prediction)
        
#run()