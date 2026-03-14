import jax
import numpy as np
from   PIL import Image
from   merkly.mtree import MerkleTree
import jax.numpy as jnp

# THRML imports (from our previous GeneNode/GeneFactor setup)
from   thrml.block_management     import Block
from   thrml.block_sampling       import BlockGibbsSpec, FactorSamplingProgram, SamplingSchedule, sample_states
from   thrml.interaction          import InteractionGroup
from   thrml.conditional_samplers import AbstractConditionalSampler
from   thrml.factor               import AbstractFactor
import equinox as eqx

# Example function inputs and desired output
function_inputs = [4, -2, 3.5, 5, -11, -4.7]

# Hash function for MerkleTree
mhash_function = lambda x, y: x + y
bytes_inputs = [bytes(str(g), "utf-8") for g in function_inputs]
mtree = MerkleTree(bytes_inputs, mhash_function)

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

def generate_thrml_from_mtree(mtree: MerkleTree):
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

    class GraphWrapper:
        def __init__(self, nodes, program):
            self.nodes   = nodes
            self.program = program

    return GraphWrapper(nodes, program)

def predict_ga_from_graph(graph, inputs):
    key = jax.random.PRNGKey(0)
    schedule = SamplingSchedule(n_warmup=5, n_samples=1, steps_per_sample=1)
    init_state = [jnp.zeros((1, len(graph.nodes)))]
    samples = sample_states(key, graph.program, schedule, init_state)
    gene_values = samples[0][0]
    prediction = sum([g * x for g, x in zip(gene_values, inputs)])
    return prediction

# --- Deterministic Run ---
def run(target_image_path):
    input_image  = []
    target_image = Image.open(target_image_path).convert("L") #"eg: target.jpg"
    num_genes    = len(function_inputs)

    for iteration in range(1, 2):  # single deterministic iteration
        graph = generate_thrml_from_mtree(mtree)
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

run()