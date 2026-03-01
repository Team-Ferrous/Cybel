import os
import json
from chatbot_llm import Chatbot

# Path to the memories folder
MEMORIES_FOLDER = "memories"

def load_memories(folder_path):
    memories = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            with open(os.path.join(folder_path, filename), "r") as file:
                memories.append(json.load(file))
    return memories

if __name__ == "__main__":
    # Load memories
    memories = load_memories(MEMORIES_FOLDER)
    
    # Initialize chatbot with memories
    chatbot = Chatbot(memories)
    
    # Start chatbot interaction
    chatbot.run()
