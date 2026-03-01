from chatbot_llm import Chatbot
#from config import BOT_NAME

BOT_NAME = "BasicBot"
def main():
    print(f"Welcome to {BOT_NAME}!")
    bot = Chatbot()
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        response = bot.get_response(user_input)
        print(f"{BOT_NAME}: {response}")

if __name__ == "__main__":
    main()
