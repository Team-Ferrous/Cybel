import os
import json 

BOT_NAME = "Mokima"
BASE_DIR = r"C:\Users\EnocEscalona\Documents\Supercell_AI_Dev (B)"
PERSONALITY_FILE = os.path.join(BASE_DIR, "Personality", "src", "character", "RynneRoyale.json")
CLEANED_JSON_DIR = os.path.join(BASE_DIR, "TARS", "src", "supercell_ai", "Cleaned_JSONs")

class Chatbot:
    def __init__(self):
        ''' self.responses = {
            "hello": "Hi there! How can I help you?",
            "how are you": "I'm just a bot, but I'm doing great! How about you?",
            "bye": "Goodbye! Have a great day!"
        }
        '''
        self.memory_file = os.path.join(CLEANED_JSON_DIR, "supercellmemory.json")
        self.memory      = self._load_memory()

    # Standalone function to test the bot with.
    def chatbot_loop(char_name, generate_response):
        """Run the chatbot loop for user interaction."""
        print(f"\n{char_name} is ready! Type your message (or 'quit' to exit).")
        while True:
            try:
                user_input = input("You: ")
                if user_input.lower() == 'quit':
                    print("Goodbye!")
                    break
                response = generate_response(user_input)
                print(f"{char_name}: {response}")
            except Exception as e:
                print(f"Error in chatbot loop: {e}")

    def _load_memory(self):
        """Load memory from the supercellmemory JSON file."""
        if os.path.exists(self.memory_file):
            with open(self.memory_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []

    def _save_memory(self):
        """Save memory to the supercellmemory JSON file."""
        with open(self.memory_file, 'w', encoding='utf-8') as f:
            json.dump(self.memory, f, ensure_ascii=False, indent=4)

    def get_response(self, user_input):
        """Generate a response from the chatbot and store the interaction in memory."""
        try:
            result   = self.qa_chain({"query": user_input})
            response = result["result"]

            # Store the interaction in memory
            self.memory.append({"user": user_input, "bot": response})
            self._save_memory()

            return response
        except Exception as e:
            return f"{self.char_name}: I'm sorry, I couldn't process your request."

