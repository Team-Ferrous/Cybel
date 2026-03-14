import os
import jax
import uuid
import hashlib
from   pydantic          import (Field, PositiveInt)
from   typing            import Optional, List, Any, TypedDict
from   typing_extensions import override

import jax.numpy as jnp
import equinox   as eqx

from langchain_core.pydantic_v1     import BaseModel, Field
from langchain_core.prompts         import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages.base   import get_msg_title_repr
from langchain_core.utils           import get_colored_text
from langchain_core.prompt_values   import ChatPromptValue, ImageURL
from langchain_core.prompts.base    import BasePromptTemplate
from langchain_core.prompts.dict    import DictPromptTemplate
from langchain_core.prompts.image   import ImagePromptTemplate
from langchain_core.prompts.message import (
    BaseMessagePromptTemplate,
)
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolMessage,
    convert_to_messages,
)

from thrml.pgm              import AbstractNode
from thrml.block_management import Block
from thrml.block_sampling   import (
    BlockGibbsSpec,
    FactorSamplingProgram,
    SamplingSchedule,
    sample_states
)
from thrml.factor               import AbstractFactor
from thrml.interaction          import InteractionGroup
from thrml.conditional_samplers import AbstractConditionalSampler


#How this plugs into Cybel
'''Example Usage

This example data mimics a GraphRAG output.
graph = {
    "nodes": ["scrooge", "bob", "tim"],
    "edges": [
        ("scrooge","bob"),
        ("bob","tim")
    ]
}

# you then simply feed the graph to thrml to get a TGraph, run tests on the TG, then query probability states:
thrml_graph = generate_thrml(graph)
samples = test_thrml(thrml_graph)
p = query_probability(thrml_graph, samples, "tim")
print("P(Tim positive state):", p)
'''

#You already have:
#grag = generate_grag(...)

#Now you could do:
#graph = grag.export_graph()
#thrml_graph = generate_thrml(graph)
#samples = test_thrml(thrml_graph)
#print(query_probability(thrml_graph, samples, "TinyTim"))

# -----------------------------
# utility
# -----------------------------

def get_sha256_hash(text: str):
    return hashlib.sha256(text.encode()).hexdigest()


class Example(TypedDict):
    """A representation of an example consisting of text input and expected tool calls.

    For extraction, the tool calls are represented as instances of pydantic model.
    """

    input: str  # This is the example text
    tool_calls: List[BaseModel]  # Instances of pydantic model that should be extracted


def tool_example_to_messages(example: Example) -> List[BaseMessage]:
    """Convert an example into a list of messages that can be fed into an LLM.

    This code is an adapter that converts our example to a list of messages
    that can be fed into a chat model.

    The list of messages per example corresponds to:

    1) HumanMessage: contains the content from which content should be extracted.
    2) AIMessage: contains the extracted information from the model
    3) ToolMessage: contains confirmation to the model that the model requested a tool correctly.

    The ToolMessage is required because some of the chat models are hyper-optimized for agents
    rather than for an extraction use case.
    """
    messages: List[BaseMessage] = [HumanMessage(content=example["input"])]
    tool_calls = []
    for tool_call in example["tool_calls"]:
        tool_calls.append(
            {
                "id": str(uuid.uuid4()),
                "args": tool_call.dict(),
                # The name of the function right now corresponds
                # to the name of the pydantic model
                # This is implicit in the API right now,
                # and will be improved over time.
                "name": tool_call.__class__.__name__,
            },
        )
    messages.append(AIMessage(content="", tool_calls=tool_calls))
    tool_outputs = example.get("tool_outputs") or [
        "You have correctly called this tool."
    ] * len(tool_calls)
    for output, tool_call in zip(tool_outputs, tool_calls):
        messages.append(ToolMessage(content=output, tool_call_id=tool_call["id"]))
    return messages

class MessagesPlaceholder(BaseMessagePromptTemplate):
    """Prompt template that assumes variable is already list of messages.

    A placeholder which can be used to pass in a list of messages.

    !!! example "Direct usage"

        ```python
        from langchain_core.prompts import MessagesPlaceholder

        prompt = MessagesPlaceholder("history")
        prompt.format_messages()  # raises KeyError

        prompt = MessagesPlaceholder("history", optional=True)
        prompt.format_messages()  # returns empty list []

        prompt.format_messages(
            history=[
                ("system", "You are an AI assistant."),
                ("human", "Hello!"),
            ]
        )
        # -> [
        #     SystemMessage(content="You are an AI assistant."),
        #     HumanMessage(content="Hello!"),
        # ]
        ```

    !!! example "Building a prompt with chat history"

        ```python
        from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a helpful assistant."),
                MessagesPlaceholder("history"),
                ("human", "{question}"),
            ]
        )
        prompt.invoke(
            {
                "history": [("human", "what's 5 + 2"), ("ai", "5 + 2 is 7")],
                "question": "now multiply that by 4",
            }
        )
        # -> ChatPromptValue(messages=[
        #     SystemMessage(content="You are a helpful assistant."),
        #     HumanMessage(content="what's 5 + 2"),
        #     AIMessage(content="5 + 2 is 7"),
        #     HumanMessage(content="now multiply that by 4"),
        # ])
        ```

    !!! example "Limiting the number of messages"

        ```python
        from langchain_core.prompts import MessagesPlaceholder

        prompt = MessagesPlaceholder("history", n_messages=1)

        prompt.format_messages(
            history=[
                ("system", "You are an AI assistant."),
                ("human", "Hello!"),
            ]
        )
        # -> [
        #     HumanMessage(content="Hello!"),
        # ]
        ```
    """

    variable_name: str
    """Name of variable to use as messages."""

    optional: bool = False
    """Whether `format_messages` must be provided.

    If `True` `format_messages` can be called with no arguments and will return an empty
    list.

    If `False` then a named argument with name `variable_name` must be passed in, even
    if the value is an empty list.
    """

    n_messages: PositiveInt | None = None
    """Maximum number of messages to include.

    If `None`, then will include all.
    """

    def __init__(
        self, variable_name: str, *, optional: bool = False, **kwargs: Any
    ) -> None:
        """Create a messages placeholder.

        Args:
            variable_name: Name of variable to use as messages.
            optional: Whether `format_messages` must be provided.

                If `True` format_messages can be called with no arguments and will
                return an empty list.

                If `False` then a named argument with name `variable_name` must be
                passed in, even if the value is an empty list.
        """
        # mypy can't detect the init which is defined in the parent class
        # b/c these are BaseModel classes.
        super().__init__(variable_name=variable_name, optional=optional, **kwargs)  # type: ignore[call-arg,unused-ignore]

    def format_messages(self, **kwargs: Any) -> list[BaseMessage]:
        """Format messages from kwargs.

        Args:
            **kwargs: Keyword arguments to use for formatting.

        Returns:
            List of `BaseMessage` objects.

        Raises:
            ValueError: If variable is not a list of messages.
        """
        value = (
            kwargs.get(self.variable_name, [])
            if self.optional
            else kwargs[self.variable_name]
        )
        if not isinstance(value, list):
            msg = (
                f"variable {self.variable_name} should be a list of base messages, "
                f"got {value} of type {type(value)}"
            )
            raise ValueError(msg)  # noqa: TRY004
        value = convert_to_messages(value)
        if self.n_messages:
            value = value[-self.n_messages :]
        return value

    @property
    def input_variables(self) -> list[str]:
        """Input variables for this prompt template.

        Returns:
            List of input variable names.
        """
        return [self.variable_name] if not self.optional else []

    @override
    def pretty_repr(self, html: bool = False) -> str:
        """Human-readable representation.

        Args:
            html: Whether to format as HTML.

        Returns:
            Human-readable representation.
        """
        var = "{" + self.variable_name + "}"
        if html:
            title = get_msg_title_repr("Messages Placeholder", bold=True)
            var = get_colored_text(var, "yellow")
        else:
            title = get_msg_title_repr("Messages Placeholder")
        return f"{title}\n\n{var}"


class Activity(BaseModel):
    """Structured representation of a recommended activity."""

    title: Optional[str] = Field(
        default=None,
        description="Short title summarizing the activity"
    )

    description: Optional[str] = Field(
        default=None,
        description="One sentence explaining what the activity involves"
    )


class Lead(BaseModel):
    """A potential opportunity, idea, or actionable insight extracted from text,
       optionally linked to a document."""

    title: Optional[str] = Field(
        default=None,
        description="Short title summarizing the lead"
    )

    description: Optional[str] = Field(
        default=None,
        description="Brief explanation of the opportunity or insight"
    )

    source_name: Optional[str] = Field(
        default=None,
        description="Name or identifier of the source (e.g., PDF title, website)"
    )

    source_bytes: Optional[bytes] = Field(
        default=None,
        description="Optional raw bytes of the source file"
    )

    source_url: Optional[str] = Field(
        default=None,
        description="Optional URL where the source can be accessed"
    )

    file_type: Optional[str] = Field(
        default=None,
        description="MIME type or file extension of the source (e.g., 'application/pdf')"
    )

class LeadExtraction(BaseModel):
    """Container for extracted leads."""
    leads: List[Lead]

#1. Text + PDF lead
with open("WINGS_Investor_Deck.pdf", "rb") as f:
    pdf_bytes = f.read()

lead = Lead(
    title="Send updated demo to WINGS",
    description="Follow up with WINGS investment group using the new Ferrous build",
    source_name="WINGS Investor Deck",
    source_bytes=pdf_bytes,
    file_type="application/pdf"
)
#2. Web link lead
lead = Lead(
    title="Check out new accelerator program",
    description="Review submission requirements for TechStars Summer batch",
    source_name="TechStars Official Page",
    source_url="https://www.techstars.com/programs/summer-2026",
)

class ActivityExtraction(BaseModel):
    """Container for extracted activities."""

    activities: List[Activity]

class Data(BaseModel):
    """Extracted activities."""

    # Creates a model so that we can extract multiple entities.
    activities: List[Activity]

#2. Generic Structured Extraction Engine
#This becomes the core reusable extractor.

def build_activity_extractor(llm):
    """
    Returns a runnable extractor that converts text into structured activities.
    """

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are an expert information extraction system.

Extract activities mentioned in the text.

Rules:
- Only extract activities explicitly described.
- Each activity must have a title and description.
- If information is missing return null.
"""
            ),
            MessagesPlaceholder("examples"),
            ("human", "{text}"),
        ]
    )

    runnable = prompt | llm.with_structured_output(ActivityExtraction)

    return runnable

def build_examples(example_pairs):
    """
    Convert structured examples into message format.
    """

    messages = []

    for text, tool_call in example_pairs:
        messages.extend(
            tool_example_to_messages({
                "input": text,
                "tool_calls": [tool_call]
            })
        )

    return messages

#4. Activity Extraction Function
#Now your main function becomes simple.
def extract_activities(text, extractor, examples):
    """
    Extract activities from text.

    Returns a list of Activity objects.
    """

    result = extractor.invoke({
        "text": text,
        "examples": examples
    })

    return result.activities

def discover_entity_types(llm, memories):
    prompt = f"""
        Analyze the following memories and identify important subject categories.

        Return a list of entity types that describe the data.

        Memories:
        {memories}

        Format:
        - EntityType
        - EntityType
        """
    result = llm.invoke(prompt).content

    return [x.strip("- ").strip() for x in result.splitlines() if x.strip()]

def discover_relationships(llm, memories):
    prompt = f"""
        Identify relationships that appear between entities in these memories.

        Examples:
        Person visits Location
        Person owns Pet
        Person attends Event

        Memories:
        {memories}

        Return format:
        EntityType -> relationship -> EntityType
        """

    result = llm.invoke(prompt).content
    return result


def discover_queries(llm, memories):
    prompt = f"""
        Based on these memories, generate example questions the user might ask.

        Memories:
        {memories}

        Return 10 queries.
        """

    result = llm.invoke(prompt).content
    return result.split("\n")

# Function to extract activities from given text
def extract_activities_info(llm, prompt, text, example_messages):
    """Extract activities from given text. Returns a list of activities. Actity object contains title and description of the activity"""
    runnable = prompt | llm.with_structured_output(schema=Data)
    result = runnable.invoke({"text": text, "examples": example_messages})
    return result.activities

def generate_activities_from_graph(llm, memory_text, graph_context):

    prompt = f"""
        You are a recommendation engine.

        Use the graph context to generate activities.

        Memory:
        {memory_text}

        Relevant Entities:
        {graph_context["entities"]}

        Relationships:
        {graph_context["relationships"]}

        Generate 3 activities.

        Format:
        1. Activity Title: Description
        """

    result = llm.invoke(prompt).content
    return extract_activities_info(result)


# -----------------------------
# Node
# -----------------------------

class CybelNode(AbstractNode):
    pass

# -----------------------------
# Interaction
# -----------------------------

class EdgeInteraction(eqx.Module):
    weight: float


# -----------------------------
# Factor
# -----------------------------

class EdgeFactor(AbstractFactor):

    weight: float

    def __init__(self, weight, blocks):
        super().__init__(blocks)
        self.weight = weight

    def to_interaction_groups(self):

        return [
            InteractionGroup(
                interaction=EdgeInteraction(self.weight),
                head_nodes=self.node_groups[0],
                tail_nodes=[self.node_groups[1]],
            ),
            InteractionGroup(
                interaction=EdgeInteraction(self.weight),
                head_nodes=self.node_groups[1],
                tail_nodes=[self.node_groups[0]],
            )
        ]


# -----------------------------
# Sampler
# -----------------------------

class CybelSampler(AbstractConditionalSampler):

    def sample(self, key, interactions, active_flags, states, sampler_state, output_sd):

        bias = jnp.zeros(output_sd.shape)

        for interaction, state in zip(interactions, states):

            if isinstance(interaction, EdgeInteraction):

                if len(state) > 0:
                    neighbor = jnp.stack(state, -1)
                    bias += interaction.weight * jnp.sum(neighbor, axis=-1)

        noise = jax.random.normal(key, output_sd.shape)

        return bias + noise, sampler_state

    def init(self):
        return None


# -----------------------------
# Graph container
# -----------------------------

class THRMLGraph:
    def __init__(self, working_dir, nodes, program, node_index):
        self.working_dir = working_dir
        self.nodes = nodes
        self.program = program
        self.node_index = node_index


# -----------------------------
# MAIN builder
# -----------------------------
def generate_thrml(graph):

    """
    graph format:
    {
        "nodes": ["scrooge", "bob", "tim"],
        "edges": [
            ("scrooge","bob"),
            ("bob","tim")
        ]
    }
    """

    input_hash = get_sha256_hash(str(graph))
    working_dir = f"./thrml_{input_hash}"

    os.makedirs(working_dir, exist_ok=True)

    # create nodes
    node_objs = {n: CybelNode() for n in graph["nodes"]}

    node_list = list(node_objs.values())
    node_index = {n: i for i, n in enumerate(graph["nodes"])}

    block = Block(node_list)

    node_shape_dtypes = {
        CybelNode: jax.ShapeDtypeStruct((), jnp.float32)
    }

    spec = BlockGibbsSpec(
        free_blocks=[block],
        clamped_blocks=[],
        node_shape_dtypes=node_shape_dtypes
    )

    factors = []

    for src, dst in graph["edges"]:

        factors.append(
            EdgeFactor(
                weight=1.0,
                blocks=(Block([node_objs[src]]), Block([node_objs[dst]]))
            )
        )

    sampler = CybelSampler()

    program = FactorSamplingProgram(
        gibbs_spec=spec,
        samplers=[sampler],
        factors=factors,
        other_interaction_groups=[]
    )

    return THRMLGraph(working_dir, node_objs, program, node_index)

#Sampling / Query Interface
def test_thrml(thrml_graph, steps=1000):

    key = jax.random.key(0)

    schedule = SamplingSchedule(
        n_warmup=20,
        n_samples=steps,
        steps_per_sample=2
    )

    init_state = [
        jax.random.normal(key, (1, len(thrml_graph.nodes)))
    ]

    samples = sample_states(
        key,
        thrml_graph.program,
        schedule,
        init_state
    )

    return samples

#Simple Probability Query
def query_probability(thrml_graph, samples, node_name):

    idx = thrml_graph.node_index[node_name]

    values = samples[0][:, idx]

    prob_positive = (values > 0).mean()

    return prob_positive
