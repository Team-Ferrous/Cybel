import json
import time
import uuid
import paho.mqtt.client as mqtt
from transformers import pipeline

# Setup local LLM pipeline (can be changed to any local model)
llm = pipeline("text-generation", model="distilgpt2")

MQTT_BROKER = "localhost"
REQUEST_TOPIC = "llm/requests"

def on_connect(client, userdata, flags, rc):
    print(f"Connected with result code {rc}")
    client.subscribe(REQUEST_TOPIC)

def on_message(client, userdata, msg):
    try:
        data = json.loads(msg.payload.decode())
        prompt = data["prompt"]
        request_id = data["request_id"]
        response_topic = f"llm/responses/{request_id}"
        print(f"Received prompt: {prompt}")

        # Run model inference
        output = llm(prompt, max_length=200, num_return_sequences=1)[0]["generated_text"]

        # Publish response
        response = {"request_id": request_id, "response": output}
        client.publish(response_topic, json.dumps(response))
        print(f"Published response to {response_topic}")

    except Exception as e:
        print(f"Error handling message: {e}")

client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

client.connect(MQTT_BROKER, 1883, 60)
client.loop_forever()
