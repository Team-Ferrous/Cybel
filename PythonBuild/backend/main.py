import os
import logging
import argparse

import weaviate
from   weaviate.classes.init   import Auth
import weaviate.classes as wvc

from dotenv import load_dotenv

#from dia import model as Dia
#from playsound import playsound
from automat_llm.core   import load_json_as_documents, load_personality_file, init_interactions, generate_response, create_rag_chain
from automat_llm.config import load_config, save_config, update_config
from rich.panel    import Panel
from rich.markdown import Markdown
from rich.console import Console
from datetime import datetime
import json

console     = Console()
config      = load_config()
current_dir = os.getcwd()

def render_llm(text: str, title: str = "LLM Response"):
    md = Markdown(text, code_theme="monokai", hyperlinks=True)
    panel = Panel(
        md,
        title=title,
        border_style="cyan",
        padding=(1, 2)
    )
    console.print(panel)

load_dotenv()

weaviate_url     = os.environ.get("WEAVIATE_URL") 
weaviate_api_key = os.environ.get("WEAVIATE_API_KEY")
user_id          = "Automat-User-Id" # config["default_user"]  # , In the future this will be in a config the user can set.
                                     # It is made for a single-user system; can be modified for multi-user

# Ensure directories exist
directory = os.path.abspath(f'{current_dir}/Input_JSON/')
if not os.path.exists(directory):
    print(f"Cleaned JSON directory not found at {directory}. Creating Input_JSON folder")
    os.mkdir(f'{current_dir}/Input_JSON')
    print("UnhandledException: please load Cleaned_, or Cybel Memory JSON into Input_JSON")
    exit()



def chat_once(user_input: str):
    try:
        response = generate_response(user_id, user_interactions, user_input, rude_keywords, personality_data, rag_chain)
        return response
    except Exception as e:
        return f"Error generating response: {e}"


def upload_logs_to_weaviate(client, log_filepath: str, state_file: str = "upload_state.json"):
    """Parse chat logs and upload only new entries to Weaviate."""
    if not os.path.exists(log_filepath):
        print(f"Log file not found at {log_filepath}")
        return
    
    # Load the last upload position
    last_position = 0
    if os.path.exists(state_file):
        try:
            with open(state_file, 'r') as f:
                state = json.load(f)
                last_position = state.get('last_line', 0)
        except:
            last_position = 0
    
    try:
        with open(log_filepath, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
        
        total_lines = len(lines)  
        
        # Only process lines after the last position
        new_lines = lines[last_position:]
        
        if not new_lines:
            print("No new logs to upload.")
            return
        
        # Parse logs (same as before, but on new_lines)
        conversations = []
        current_user_input = None
        
        for line in new_lines:
            line = line.strip()
            if not line:
                continue
            
            if " - INFO - User: " in line:
                user_part = line.split(" - INFO - User: ", 1)
                if len(user_part) == 2:
                    current_user_input = user_part[1].strip()
            
            elif " - INFO - Bot: " in line and current_user_input:
                bot_part = line.split(" - INFO - Bot: ", 1)
                if len(bot_part) == 2:
                    bot_response = bot_part[1].strip()
                    
                    conversations.append({
                        "user_input": current_user_input,
                        "bot_response": bot_response
                    })
                    current_user_input = None
        
        if not conversations:
            print("No new complete conversations found.")

            with open(state_file, 'w') as f:
                json.dump({'last_line': total_lines}, f)
            return
        
        print(f"Found {len(conversations)} new conversations to upload.")
        
        # Upload to Weaviate
        collection = client.collections.get("SampleData")
        
        uploaded_count = 0
        with collection.batch.dynamic() as batch:
            for conv in conversations:
                conversation_text = f"""User: {conv['user_input']}
Bot: {conv['bot_response']}"""
                
                batch.add_object(
                    properties={
                        "text": conversation_text,
                        "metadata": json.dumps({
                            "type": "conversation_log",
                            "source": "chatbot_logs",
                            "uploaded_at": datetime.now().isoformat()
                        })
                    }
                )
                uploaded_count += 1
        

        with open(state_file, 'w') as f:
            json.dump({'last_line': total_lines}, f)
        
        print(f"✅ Successfully uploaded {uploaded_count} new conversations to Weaviate.")
        
    except Exception as e:
        print(f"❌ Error uploading logs: {e}")
        import traceback
        traceback.print_exc()
        

def check_weaviate_contents(client):
    """Check what's actually in Weaviate."""
    try:
        collection = client.collections.get("SampleData")
        
        # Get total count
        result = collection.aggregate.over_all(total_count=True)
        total = result.total_count
        
        print(f"\n{'='*50}")
        print(f"📊 SampleData Collection Statistics")
        print(f"{'='*50}")
        print(f"Total objects: {total}")
        
        if total > 0:
            # Fetch first 5 objects to see what they look like
            print(f"\n Sample objects:")
            for i, item in enumerate(collection.iterator()):
                if i >= 5: break
                print(f"\nObject {i+1}:")
                print(f"  UUID: {item.uuid}")
                print(f"  Properties: {item.properties}")
        else:
            print("\n⚠️  Collection is empty!")
            
    except Exception as e:
        print(f"Error checking Weaviate: {e}")
        import traceback
        traceback.print_exc()


def initialize_system(weaviate_client):
    global client, personality_data, user_interactions, documents, rag_chain, rude_keywords

    client = weaviate_client

    # if os.environ.get("LOCAL_WEAVIATE") == "1":
    #     client = weaviate.connect_to_local(
    #         additional_config=wvc.init.AdditionalConfig(
    #             timeout=wvc.init.Timeout(init=60, query=30, insert=120)
    #         )
    #     )
    # else:
    #     client = weaviate.connect_to_weaviate_cloud(
    #         cluster_url=weaviate_url,
    #         auth_credentials=Auth.api_key(weaviate_api_key),
    #     )

    personality_data = load_personality_file()
    user_interactions = init_interactions()
    documents = load_json_as_documents(client, directory)
    
    if not documents:
        print("No documents loaded. Exiting.")
        exit()

    print(f"Loaded {len(documents)} documents for RAG.")

    rude_keywords = ["stupid", "idiot", "shut up", "useless", "dumb"]
    rag_chain = create_rag_chain(client, user_id, documents)

    check_weaviate_contents(client)


# Set up logging to save chatbot interactions
logging.basicConfig(
    filename=f'{current_dir}/Logs/chatbot_logs.txt', #r'./Logs/chatbot_logs.txt',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Chatbot loop
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo of boolean flag with argparse.")
    parser.add_argument("--set", metavar="KEY=VALUE", help="Set a configuration value (e.g., user.name=Alice)")
    parser.add_argument("--use_dia",          action="store_true",  help="Enable Dia audio model use and output") # Boolean flag
    parser.add_argument("--local_connection", action="store_true", help="Use Weaviate locally via docker-compose.") # Boolean flag
    args = parser.parse_args()

    if args.use_dia:
       dia_model = Dia.from_pretrained("nari-labs/Dia-1.6B", compute_dtype="float16")
       print("Audio mode is ON")
    else:
       print("Audio mode is OFF")

    if args.local_connection == True:
        client = weaviate.connect_to_local(additional_config=wvc.init.AdditionalConfig(timeout=wvc.init.Timeout(init=60,query=30, insert=120)))
    else:
        client = weaviate.connect_to_weaviate_cloud(
        cluster_url=weaviate_url,                                    
       auth_credentials=Auth.api_key(weaviate_api_key),              
    )

    personality_data  = load_personality_file()
    user_interactions = init_interactions()
    documents         = load_json_as_documents(client, directory)

    if not documents:
        print("No documents extracted from JSON files. Please check the file contents.")
        exit()

    print(f"Loaded {len(documents)} documents for RAG.")

    # Extract personality details
    char_name     = personality_data['char_name']

    # Rudeness detection keywords
    rude_keywords = ["stupid", "idiot", "shut up", "useless", "dumb"]
    rag_chain     = create_rag_chain(client, user_id, documents)

    if args.set:
        if "=" not in args.set:
            parser.error("Argument to --set must be in key=value format.")
        key, value = args.set.split("=", 1)

        # Type inference (primitive)
        if value.lower() in ("true", "false"):
            value = value.lower() == "true"
        elif value.isdigit():
            value = int(value)

        config = update_config(config, key, value)
        save_config(config)
        print(f"Updated {key} to {value}")

    print(f"Current config:\n{config}")
    print(f"\n{char_name} is ready! Type your message (or 'quit' to exit).")
    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() == 'quit':
                print("Goodbye!")
                break
            if user_input.__contains__('image'):
                generator.generate_image(user_input, f"newbie_sample_{len(user_interactions)}")
                break
            response = generate_response(user_id, user_interactions, user_input, rude_keywords, personality_data, rag_chain)
            #if(args.use_dia):
                #output = dia_model.generate(f"[S1] {response}", use_torch_compile=True, verbose=True)
                #dia_model.save_audio(f"response.mp3", output)
                #playsound("response.mp3")
            #print(f"{char_name}: {response}")
            render_llm(response, title=f"{char_name}")
        except Exception as e:
            print(f"Error in chatbot loop: {e}")