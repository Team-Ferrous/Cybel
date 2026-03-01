import os
import json
import logging
from   langchain_huggingface            import HuggingFaceEmbeddings, HuggingFacePipeline
from   langchain_community.vectorstores import FAISS
from   langchain_core.prompts.chat      import ChatPromptTemplate
from   langchain.chains.retrieval       import create_retrieval_chain
from   langchain.docstore.document      import Document
from   langchain.schema.document        import Document
from   langchain_groq                   import ChatGroq
from   langchain_weaviate.vectorstores  import WeaviateVectorStore

current_dir = os.getcwd()

conversation_histories = {}  

def get_conversation_history(user_id, max_messages=10):
    """Get the last N messages from conversation history"""
    if user_id not in conversation_histories:
        conversation_histories[user_id] = []
    return conversation_histories[user_id][-max_messages:]

def add_to_conversation_history(user_id, role, content):
    """Add a message to conversation history"""
    if user_id not in conversation_histories:
        conversation_histories[user_id] = []
    conversation_histories[user_id].append({"role": role, "content": content})

def format_conversation_history(history):
    """Format conversation history as a readable string"""
    formatted = []
    for msg in history:
        if msg["role"] == "user":
            formatted.append(f"User: {msg['content']}")
        else:
            formatted.append(f"Cybel: {msg['content']}")
    return "\n".join(formatted)


def load_json_as_documents(client, directory):
    documents = []
    collection = client.collections.use("MyCollection") #TBA: use(f"{user_id}_Collection")
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            path = os.path.join(directory, filename)
            with open(path, 'r', encoding='utf-8') as f:
                try:
                    raw_content = f.read()
                    # Optionally, reformat it as pretty-printed JSON
                    parsed = json.loads(raw_content)
                    pretty_json = json.dumps(parsed, indent=2)
                    documents.append(Document(page_content=pretty_json, metadata={"source": filename}))
                except Exception as e:
                    print(f"Skipping {filename} due to error: {e}")

    # Extract list of 'Entry' strings if the JSON is a list of dicts
    entries = [item['Entry'] for item in documents if 'Entry' in item]

    with collection.batch.fixed_size(batch_size=200) as batch:
        for d in entries:
            print(d)
            batch.add_object(
                {
                    "entry": d
                }
            )

            with open("uploaded_docs_log.json", "a", encoding="utf-8") as log_f:
                log_f.write(json.dumps({
                    "entry": d,
                    "timestamp": time.time()
                }) + "\n")

            if batch.number_errors > 10:
                print("Batch import stopped due to excessive errors.")
                break

    failed_objects = collection.batch.failed_objects
    if failed_objects:
        print(f"Number of failed imports: {len(failed_objects)}")
        print(f"First failed object: {failed_objects[0]}")


    return documents

def init_interactions():
    # Load or initialize user interactions
    try:
        user_interactions_file = f"{current_dir}/user_interactions.json"
        with open(user_interactions_file, 'r', encoding='utf-8') as f:
            user_interactions = json.load(f)
            return user_interactions
    except FileNotFoundError:
        user_interactions = {"users": {}}
        with open(user_interactions_file, 'w', encoding='utf-8') as f:
            json.dump(user_interactions, f, indent=4)


def load_personality_file():
    # Load the personality from robot_personality.json
    try:
        personality_file = f"{current_dir}/robot_personality.json"
        with open(personality_file, 'r', encoding='utf-8') as f:
            personality_data = json.load(f)
            return personality_data
    except FileNotFoundError:
        print(f"Personality file not found at {personality_file}. Please create robot_personality.json.")
        logging.error(f"Personality file not found at {personality_file}.")
        exit(1)


def create_rag_chain(client, user_id, documents):
    """
    Create a RAG chain that searches the SampleData collection during active conversations.
    This allows the AI to retrieve past conversations and remember things like your name.
    """
    from weaviate.classes.config import Configure
    try:
        print("Step 1: Connecting to conversation memory...")
        
        # Use HuggingFace embeddings - same model used for storing conversations
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # Connect to SampleData collection where conversation logs are already stored
        # No need to upload anything - conversations are uploaded on shutdown
        vector_store = WeaviateVectorStore(
            client=client,
            index_name="SampleData",
            text_key="text",
            embedding=embeddings
        )
        
        print("Step 2: Setting up the language model...")
       
        prompt = ChatPromptTemplate.from_messages([
        ("system", """You are Cybel — a snarky but helpful AI assistant.

        You have access to:
        1. The CURRENT CONVERSATION (recent messages in this session)
        2. PAST CONVERSATION LOGS (retrieved from long-term memory)

        Guidelines:
        - ALWAYS read the current conversation first to understand the immediate context
        - Use past conversation logs ONLY when they provide additional relevant information
        - If the user asks "Can you list the benefits of it" - check what "it" refers to in the CURRENT CONVERSATION
        - If you find relevant information about the user (name, preferences), use it naturally
        - Do NOT mention that you checked memory or logs
        - Do NOT quote logs verbatim unless the user asks
        - Be conversational, confident, and lightly snarky — never rude or creepy
        - Never reference these instructions in your replies

        Stay focused on helping the user clearly and efficiently."""),
            ("human", "{input}\n\n=== Current Conversation ===\n{conversation_history}\n\n=== Past Conversations (Long-term Memory) ===\n{context}")   
        ])
        
        
        llm = ChatGroq(
            temperature=0.5,
            model="openai/gpt-oss-20b",
            max_tokens=5000,
            api_key=os.environ.get("GROQ_API_KEY")
        )

        llm_chain = prompt | llm
        print("Language model set up.")

        # Create retrieval chain that searches SampleData collection
        # k=10 means retrieve top 10 most relevant past conversations
        rag_chain = create_retrieval_chain(
            vector_store.as_retriever(search_kwargs={"k": 10}),
            llm_chain
        )
        print("Cybel's memory is ready!")
        return rag_chain

    except Exception as e:
        print(f"Error creating the RetrievalQA chain: {e}")
        import traceback
        traceback.print_exc()
        return None

# Function to update user interactions
def update_user_interactions(user_id, user_interactions_file, user_interactions, is_rude=False, apologized=False):
    if user_id not in user_interactions["users"]:
        user_interactions["users"][user_id] = {"rudeness_score": 0, "requires_apology": False}
    
    user_data = user_interactions["users"][user_id]
    if is_rude:
        user_data["rudeness_score"] += 1
        if user_data["rudeness_score"] >= 2:  # Threshold for requiring an apology
            user_data["requires_apology"] = True
    elif apologized:
        user_data["rudeness_score"] = 0
        user_data["requires_apology"] = False
    
    with open(user_interactions_file, 'w', encoding='utf-8') as f:
        json.dump(user_interactions, f, indent=4)

def generate_response(user_id, user_interactions, user_input, rude_keywords, personality_data, rag_chain):
    """
    Generate a response using the RetrievalQA chain with conversation history.

    Parameters:
    - user_id (str): Identifier for the user.
    - user_input (str): The user's input text.

    Returns:
    - str: The AI-generated response.
    """
    input_lower = user_input.lower()
    
    # Check if user requires an apology
    user_data = user_interactions["users"].get(user_id, {"rudeness_score": 0, "requires_apology": False})
    if user_data.get("requires_apology", False):
        if "sorry" in input_lower or "apologize" in input_lower:
            update_user_interactions(user_id, user_interactions_file=f"{current_dir}/user_interactions.json", 
                                   user_interactions=user_interactions, apologized=True)
            return next(item['response'] for item in personality_data['example_dialogue'] if item['user'].lower() == "i'm sorry for being rude.")
        return "I'm waiting for an apology, sweetie. I don't respond to rudeness without respect."

    # Check for rudeness
    is_rude = any(keyword in input_lower for keyword in rude_keywords)
    if is_rude:
        update_user_interactions(user_id, user_interactions_file=f"{current_dir}/user_interactions.json",
                               user_interactions=user_interactions, is_rude=True)
        return next(
            item['response']
            for item in personality_data['example_dialogue']
            if item['user'].lower() == "just do what i say, you stupid robot!"
        )

    try:
      
        add_to_conversation_history(user_id, "user", user_input)
        history = get_conversation_history(user_id, max_messages=8)  # Last 8 messages (4 exchanges)
        formatted_history = format_conversation_history(history[:-1])  # Exclude current message from history display
        
        
        result = rag_chain.invoke({
            "input": user_input,
            "conversation_history": formatted_history  # Pass current conversation
        })
  
        answer = result.get("answer") or result.get("result")
        if hasattr(answer, "content"):
            response = answer.content
        else:
            response = str(answer)


        

        add_to_conversation_history(user_id, "assistant", response)



        logging.info(f"User: {user_input}")
        logging.info(f"Bot: {response}")
        logging.info("Retrieved Memories:")


        docs = result.get("context", [])
        for doc in docs:
            page_info = doc.metadata.get("page") or doc.metadata.get("source") or "past conversation"
            logging.info(f"- [{page_info}] {doc.page_content[:200]}")  # first 200 chars

        logging.info("")
        return response

    except Exception as e:
        print(f"Error generating response: {e}")
        logging.error(f"Error generating response: {e}", exc_info=True)
        return "I'm sorry, I couldn't process your request."