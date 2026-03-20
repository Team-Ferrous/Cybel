# sparc_server.py
from thrml_merkle_pygad import build_mtree, run_image as _run_image, run_text as _run_text
from fast_graphrag import generate_grag
from oqtopus_graph_generator import QuantumCircuitGraph, QuantumNode, extract_semantic_graph
from thrml_graph_generator import discover_entity_types, discover_relationships, discover_queries, generate_thrml
from thrml_merkle_pygad import generate_thrml_from_mtree, predict_ga_from_graph
from oqtopus_merkle_pygad import build_merkle_quantum_graph, run_merkle_quantum_query

import sys

# Only these functions are callable via JS
_allowed_commands = {
    "run-image": _run_image,
    "run-text":  _run_text,
    "generate-grag": lambda text: generate_grag(text),
    "extract-semantic-graph": lambda text: extract_semantic_graph(None, text),  # LLM integration would go here
    "discover-entity-types": lambda text: discover_entity_types(text),
    "discover-relationships": lambda text: discover_relationships(text),
    "discover-queries": lambda text: discover_queries(text),
    "generate-mtree": lambda inputs: build_mtree(inputs),
    "generate-thrml-graph": lambda text: generate_thrml(text),  # Placeholder for actual graph generation
    "generate-quantum-graph": lambda _: QuantumCircuitGraph(
        nodes=[QuantumNode(f"q{i}") for i in range(5)],
    ),
}

def main():
    args = sys.argv[1:]

    # Determine which command
    command = None
    param   = None

    if "--run-image" in args:
        command = "run-image"
        param = args[args.index("--run-image") + 1]

    elif "--run-text" in args:
        command = "run-text"
        param = args[args.index("--run-text") + 1]

    else:
        print("No valid command provided. Allowed: --run-image <path>, --run-text <text>")
        sys.exit(1)

    # Execute only allowed commands
    func = _allowed_commands.get(command)
    if func:
        func(param)
    else:
        print(f"Command {command} is not allowed.")
        sys.exit(1)

if __name__ == "__main__":
    main()
