import os
import hashlib
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

import equinox as eqx
from thrml.factor import AbstractFactor
from thrml.interaction import InteractionGroup


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

from thrml.conditional_samplers import AbstractConditionalSampler

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

def test_thrml(graph:THRMLGraph, steps=500):

    key = jax.random.key(0)

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