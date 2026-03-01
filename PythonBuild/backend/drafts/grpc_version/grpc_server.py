import grpc
from concurrent import futures
import threading  # Add threading for gRPC server
import example_pb2_grpc
import example_pb2
import agent_pb2
import agent_pb2_grpc
from langchain.llms import OpenAI  # Example LLM agent from LangChain
from flask import Flask, request, jsonify

app = Flask(__name__)

# gRPC Server Code
class ExampleServiceServicer(example_pb2_grpc.ExampleServiceServicer):
    def SendRequest(self, request, context):
        reply_message = f"Received: {request.message}"
        return example_pb2.ExampleResponse(reply=reply_message)

# Implement the gRPC service
class AIServiceServicer(agent_pb2_grpc.AIServiceServicer):
    def __init__(self):
        # Initialize the LangChain agent (for example, using OpenAI LLM)
        self.llm = OpenAI(model_name="text-davinci-003", openai_api_key="YOUR_API_KEY")

    def AskAgent(self, request, context):
        # Process the query using LangChain
        query = request.query
        response = self.llm(query)
        return agent_pb2.AgentResponse(answer=response)

def start_grpc_server():
    # Load the server certificate and private key for TLS/SSL
    with open('server_cert.pem', 'rb') as f:
        server_cert = f.read()
    with open('server_key.pem', 'rb') as f:
        server_key = f.read()

    # Create server credentials for TLS
    server_credentials = grpc.ssl_server_credentials(((server_key, server_cert,),))

    # Create a gRPC server and add services
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    agent_pb2_grpc.add_AIServiceServicer_to_server(AIServiceServicer(), server)
    example_pb2_grpc.add_ExampleServiceServicer_to_server(ExampleServiceServicer(), server)

    # Add a secure port and start the gRPC server
    server.add_secure_port('[::]:50051', server_credentials)
    server.start()
    print("gRPC Server started on port 50051 with TLS")
    server.wait_for_termination()

# gRPC Client Code
def send_grpc_request(message):
    # Load the client certificate and root certificate
    with open('client_cert.pem', 'rb') as f:
        client_cert = f.read()
    with open('root_cert.pem', 'rb') as f:
        root_cert = f.read()

    # Create client credentials for mutual TLS
    credentials = grpc.ssl_channel_credentials(root_cert, client_cert)

    # Create a secure channel to the gRPC server with TLS
    with grpc.secure_channel('localhost:50051', credentials) as channel:
        stub = agent_pb2_grpc.AIServiceStub(channel)
        # Create a gRPC request message
        grpc_request = agent_pb2.AgentRequest(query=message)
        # Send the request to the gRPC server and get the response
        grpc_response = stub.AskAgent(grpc_request)
        return grpc_response.answer

# Flask API route to accept HTTP POST requests
@app.route('/ask_ai', methods=['POST'])
def ask_ai():
    try:
        # Get the query from the HTTP request body (as JSON)
        data = request.get_json()
        query = data.get('query')

        if not query:
            return jsonify({"error": "Query is required"}), 400

        # Send the query to the gRPC server using the gRPC client
        ai_response = send_grpc_request(query)

        # Return the AI response as a JSON response
        return jsonify({"response": ai_response}), 200

    except Exception as e:
        # Handle exceptions
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Start gRPC server in a separate thread
    grpc_thread = threading.Thread(target=start_grpc_server, daemon=True)
    grpc_thread.start()

    # Start Flask server
    app.run(port=5000)