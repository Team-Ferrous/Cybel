import jax
import jax.numpy as jnp
import equinox   as eqx
from thrml.pgm import AbstractNode
from thrml.block_management import Block
from thrml.factor import AbstractFactor
from thrml.interaction import InteractionGroup
from thrml.conditional_samplers import AbstractConditionalSampler
from thrml.block_sampling import BlockGibbsSpec, FactorSamplingProgram
from thrml.block_sampling import sample_states, SamplingSchedule

graph = {
    "nodes": [
        {"id": "scrooge", "type": "Character"},
        {"id": "bob", "type": "Character"},
        {"id": "tim", "type": "Character"}
    ],
    "edges": [
        ("scrooge", "employs", "bob"),
        ("bob", "parent_of", "tim")
    ]
}

class StateNode(AbstractNode):
    pass

class RelationshipInteraction(eqx.Module):
    weight: float

class RelationshipFactor(AbstractFactor):
    weight: float

    def __init__(self, weight, blocks):
        super().__init__(blocks)
        self.weight = weight

    def to_interaction_groups(self):
        return [
            InteractionGroup(
                interaction=RelationshipInteraction(self.weight),
                head_nodes=self.node_groups[0],
                tail_nodes=[self.node_groups[1]],
            )
        ]

class SimpleGaussianSampler(AbstractConditionalSampler):

    def sample(self, key, interactions, active_flags, states, sampler_state, output_sd):

        bias = 0.0
        var = 1.0

        for interaction, active, state in zip(interactions, active_flags, states):

            if isinstance(interaction, RelationshipInteraction):
                neighbor_state = jnp.stack(state, axis=-1)
                bias += interaction.weight * jnp.sum(neighbor_state)

        noise = jax.random.normal(key, output_sd.shape)

        return bias + noise * var, sampler_state

    def init(self):
        return None

    
node_objects = {}

for node in graph["nodes"]:
    node_objects[node["id"]] = StateNode()

all_nodes = list(node_objects.values())

block = Block(all_nodes)

factors = []

for src, rel, dst in graph["edges"]:
    src_node = node_objects[src]
    dst_node = node_objects[dst]

    factors.append(
        RelationshipFactor(
            weight=1.0,
            blocks=(Block([src_node]), Block([dst_node]))
        )
    )

node_shape_dtypes = {
    StateNode: jax.ShapeDtypeStruct((), jnp.float32)
}

spec = BlockGibbsSpec(
    free_blocks=[block],
    clamped_blocks=[],
    node_shape_dtypes=node_shape_dtypes
)

sampler = SimpleGaussianSampler()

program = FactorSamplingProgram(
    gibbs_spec=spec,
    samplers=[sampler],
    factors=factors,
    other_interaction_groups=[]
)

schedule = SamplingSchedule(
    n_warmup=10,
    n_samples=1000,
    steps_per_sample=2
)

key = jax.random.key(0)

init_state = [
    jax.random.normal(key, (1, len(all_nodes)))
]

samples = sample_states(
    key,
    program,
    schedule,
    init_state
)