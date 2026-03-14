import os
from typing import List, Tuple
#from PyPDF2 import PdfReader
from  pypdf import PdfReader

from langchain_groq import ChatGroq
from thrml_graph_generator import (
    generate_grag,
    discover_entity_types,
    discover_relationships,
    discover_queries,
)

import oqtopus as oq
import numpy   as np

# -----------------------------
# OQTOPUS Quantum Adapter
# -----------------------------
class QuantumNode:
    """Represents a qubit in a quantum circuit."""
    def __init__(self, name: str):
        self.name = name
        self.index = None  # will be set when added to circuit

class QuantumCircuitGraph:
    """Represents a quantum computation as a graph of qubits + gates."""
    def __init__(self, nodes: List[QuantumNode], edges: List[Tuple[str, str]]):
        self.nodes = {n.name: n for n in nodes}
        self.edges = edges  # edges represent entanglement or two-qubit gates
        self.circuit = None  # will hold OQTOPUS circuit

    def build_circuit(self):
        """Convert the graph to an OQTOPUS quantum circuit."""
        n_qubits = len(self.nodes)
        for i, node in enumerate(self.nodes.values()):
            node.index = i

        # Initialize circuit
        self.circuit = oq.QuantumCircuit(n_qubits)

        # Add simple one-qubit gates for each node
        for node in self.nodes.values():
            self.circuit.h(node.index)  # example: Hadamard gate to initialize superposition

        # Add two-qubit entangling gates based on edges
        for src_name, dst_name in self.edges:
            src = self.nodes[src_name].index
            dst = self.nodes[dst_name].index
            self.circuit.cx(src, dst)  # CNOT as example entanglement

    def run_simulation(self, shots=1024):
        """Run the circuit on a simulator and return measurement counts."""
        if self.circuit is None:
            raise ValueError("Circuit not yet built")

        sim = oq.Simulator()  # OQTOPUS simulator backend
        result = sim.run(self.circuit, shots=shots)
        counts = result.get_counts()
        return counts

# -----------------------------
# Example usage
# -----------------------------
'''if __name__ == "__main__":
    # Define qubits (nodes)
    qubits = [QuantumNode("q0"), QuantumNode("q1"), QuantumNode("q2")]

    # Define interactions (edges) - could represent entanglement
    edges = [("q0", "q1"), ("q1", "q2")]

    # Generate quantum graph
    qgraph = QuantumCircuitGraph(qubits, edges)
    qgraph.build_circuit()

    # Simulate circuit and get results
    samples = qgraph.run_simulation(shots=1000)
    print("Quantum measurement samples:", samples)

    # Query: probability that q2 = 1
    total_shots = sum(samples.values())
    prob_q2_1 = sum(count for bitstring, count in samples.items() if bitstring[-1] == "1") / total_shots
    print(f"P(q2=1): {prob_q2_1}")'''

def main_test():
    # -----------------------------
    # Initialize LLM
    # -----------------------------
    llm = ChatGroq(
        temperature=0,
        model="Llama-3.1-70b-Versatile",
        max_tokens=8000,
        api_key=os.environ.get("GROQ_API_KEY")
    )

    # -----------------------------
    # Step 1: Load PDF and convert to text
    # -----------------------------
    pdf_path = "./WINGS_Investor_Deck.pdf"
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
    DOMAIN = "Automatically extracted domain from PDF document"
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
    # Step 4: Convert GRAG → OQTOPUS Graph
    # -----------------------------
    # Map nodes to qubits
    oqt_nodes = [QuantumNode(name) for name in grag.graph["nodes"]]

    # Map edges to entanglement or controlled operations
    edges = grag.graph["edges"]  # expected [(src,dst), ...]
    oqt_graph = QuantumCircuitGraph(qubits=oqt_nodes, edges=edges)

    # Optionally encode relationships as rotation angles
    for src, dst in edges:
        weight = grag.edge_weight(src, dst) if hasattr(grag, "edge_weight") else 0.5
        oqt_graph.add_controlled_rotation(src_node=src, dst_node=dst, angle=weight)

    # Build the circuit
    oqt_graph.build_circuit()

    # -----------------------------
    # Step 5: Run quantum simulation / measure probabilities
    # -----------------------------
    shots = 1000
    samples = oqt_graph.run_simulation(shots=shots)

    # Example: query probability of first discovered entity being "active"
    query_node = entity_types[0] if entity_types else oqt_nodes[0].name
    prob_active = sum(
        count for bitstring, count in samples.items()
        if bitstring[oqt_nodes.index(next(q for q in oqt_nodes if q.name == query_node))] == "1"
    ) / shots

    print(f"\nP({query_node} active in quantum simulation):", prob_active)

def main(llm, pdf_path, DOMAIN, shots):
    # -----------------------------
    # Step 1: Load PDF and convert to text
    # -----------------------------
    #pdf_path = "./WINGS_Investor_Deck.pdf"
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
    # Step 4: Convert GRAG → OQTOPUS Graph
    # -----------------------------
    # Map nodes to qubits
    oqt_nodes = [QuantumNode(name) for name in grag.graph["nodes"]]

    # Map edges to entanglement or controlled operations
    edges = grag.graph["edges"]  # expected [(src,dst), ...]
    oqt_graph = QuantumCircuitGraph(qubits=oqt_nodes, edges=edges)

    # Optionally encode relationships as rotation angles
    for src, dst in edges:
        weight = grag.edge_weight(src, dst) if hasattr(grag, "edge_weight") else 0.5
        oqt_graph.add_controlled_rotation(src_node=src, dst_node=dst, angle=weight)

    # Build the circuit
    oqt_graph.build_circuit()

    # -----------------------------
    # Step 5: Run quantum simulation / measure probabilities
    # -----------------------------
    #shots = 1000
    samples = oqt_graph.run_simulation(shots=shots)

    # Example: query probability of first discovered entity being "active"
    query_node = entity_types[0] if entity_types else oqt_nodes[0].name
    prob_active = sum(
        count for bitstring, count in samples.items()
        if bitstring[oqt_nodes.index(next(q for q in oqt_nodes if q.name == query_node))] == "1"
    ) / shots

    print(f"\nP({query_node} active in quantum simulation):", prob_active)