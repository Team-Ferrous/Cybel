import os
import json
from langchain.docstore.document import Document

def load_json_files(directory):
    """Load and validate JSON files from a directory."""
    json_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.json')]
    documents = []
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                first_char = f.read(1)
                f.seek(0)
                if first_char == '[':
                    data = json.load(f)
                    documents.extend([Document(page=entry['text']) for entry in data if 'text' in entry])
                else:
                    for line in f:
                        entry = json.loads(line.strip())
                        if 'text' in entry:
                            documents.append(Document(page=entry['text']))
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
    return documents

def load_json_file(file_path):
    documents = []
    with open(file_path, 'r', encoding='utf-8') as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char == '[':
            data = json.load(f)
            documents.extend([Document(page_content=entry['text']) for entry in data if 'text' in entry])
        else:
            for line in f:
                entry = json.loads(line.strip())
                if 'text' in entry:
                    documents.append(Document(page_content=entry['text']))
    return documents
