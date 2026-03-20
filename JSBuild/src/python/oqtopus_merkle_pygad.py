from typing import List, Tuple
from merkly.mtree import MerkleTree
from oqtopus_graph_generator import QuantumNode, QuantumCircuitGraph

# -----------------------------
# Extend QuantumNode for Merkle
# -----------------------------
class MerkleQuantumNode(QuantumNode):
    def __init__(self, name: str, data_bytes: bytes):
        super().__init__(name)
        self.data_hash = data_bytes  # store leaf bytes for verification

# -----------------------------
# Build Quantum Graph from Merkle Tree
# -----------------------------
def build_merkle_quantum_graph(inputs: List[int], edges: List[Tuple[str, str]]):
    # Convert inputs to bytes
    bytes_inputs = [bytes(str(x), "utf-8") for x in inputs]
    mhash_function = lambda x, y: x + y
    mtree = MerkleTree(bytes_inputs, mhash_function)

    # Create nodes from Merkle leaves
    nodes = [
        MerkleQuantumNode(f"q{i}", leaf)
        for i, leaf in enumerate(mtree.leaves)
    ]

    # Create the graph
    qgraph = QuantumCircuitGraph(nodes, edges)

    # Optionally attach Merkle tree to the graph for verification
    qgraph.mtree = mtree
    return qgraph

# -----------------------------
# Example usage
# -----------------------------
def run_merkle_quantum_query(inputs, edges, shots):
    # Example inputs and edges
    # inputs = [4, -2, 3.5, 5, -11, -4.7]
    # edges  = [("q0", "q1"), ("q1", "q2"), ("q2", "q3")]

    # Build Merkle-backed quantum graph
    qgraph = build_merkle_quantum_graph(inputs, edges)

    # Build circuit
    qgraph.build_circuit()

    # Simulate
    #shots = 1000
    samples = qgraph.run_simulation(shots=shots)
    print("Quantum simulation counts:", samples)

    # Optional: inspect Merkle hashes
    for node in qgraph.nodes.values():
        print(f"Node {node.name} leaf hash bytes: {node.data_hash}")

    # Access the root hash
    print("Merkle root hash:", qgraph.mtree.root)