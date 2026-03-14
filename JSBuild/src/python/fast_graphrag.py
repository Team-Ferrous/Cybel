import hashlib
from fast_graphrag import GraphRAG

# Example Input:
#   DOMAIN = "Analyze this story and identify the characters. Focus on how they interact with each other, the locations they explore, and their relationships."

#   EXAMPLE_QUERIES = [
#    "What is the significance of Christmas Eve in A Christmas Carol?",
#    "How does the setting of Victorian London contribute to the story's themes?",
#    "Describe the chain of events that leads to Scrooge's transformation.",
#    "How does Dickens use the different spirits (Past, Present, and Future) to guide Scrooge?",
#    "Why does Dickens choose to divide the story into \"staves\" rather than chapters?"
#   ]

#ENTITY_TYPES = ["Character", "Animal", "Place", "Object", "Activity", "Event"]

def get_sha256_hash(input_string):
  """
  Calculates the SHA256 hash of an input string.

  Args:
    input_string: The string to hash.

  Returns:
    The SHA256 hash as a hexadecimal string.
  """
  # Encode the string to bytes
  encoded_string = input_string.encode('utf-8')

  # Create a SHA256 hash object
  hash_object = hashlib.sha256(encoded_string)

  # Get the hexadecimal representation of the hash
  hex_digest = hash_object.hexdigest()

  return hex_digest

def generate_grag(textTitle, DOMAIN:str, ENTITY_TYPES:list[str], EXAMPLE_QUERIES:list[str]):
    input_hash = get_sha256_hash(textTitle)

    grag = GraphRAG(
        working_dir=f"./{input_hash}",
        domain=DOMAIN,
        example_queries="\n".join(EXAMPLE_QUERIES),
        entity_types=ENTITY_TYPES
    )

    #assuming document is local and is TXT, TBA: Json compatible version.
    with open(f"./{textTitle}.txt") as f:
        grag.insert(f.read())
    return grag

def test_grag(grag, q:str):
    print(grag.query(q).response)