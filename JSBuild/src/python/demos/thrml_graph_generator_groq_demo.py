import os
#from PyPDF2 import PdfReader
from   pypdf import PdfReader

from thrml_graph_generator import (
    generate_grag,
    generate_thrml,
    test_thrml,
    query_probability,
    discover_entity_types,
    discover_relationships,
    discover_queries,
)
from langchain_groq import ChatGroq

# -----------------------------
# Initialize LLM
# -----------------------------
llm = ChatGroq(
    temperature=0,
    model="Llama-3.1-70b-Versatile",
    max_tokens=8000,
    api_key=os.environ.get("GROQ_API_KEY")  # Or set manually
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
entity_types = discover_entity_types(llm, text_content)
relationships = discover_relationships(llm, text_content)
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
# Step 4: Export GraphRAG graph for THRML
# -----------------------------
graph_data = grag.export_graph()  # Expected format: {"nodes": [...], "edges": [(src,dst), ...]}
thrml_graph = generate_thrml(graph_data)

# -----------------------------
# Step 5: Sample THRML states and query probabilities
# -----------------------------
samples = test_thrml(thrml_graph, steps=500)  # Reduce steps for quick demo
query_node = entity_types[0] if entity_types else "Tim"  # Query first discovered entity or default
p = query_probability(thrml_graph, samples, query_node)

print(f"\nP({query_node} positive state):", p)