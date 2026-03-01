from fastapi import FastAPI, Request
from pydantic import BaseModel
import asyncio
import uvicorn
import json
import uuid
from gmqtt import Client as MQTTClient

app = FastAPI()

# ---------- CONFIG ----------
MQTT_BROKER_HOST = 'localhost'
MQTT_PORT = 1883
MQTT_TOPIC_REQUEST = 'robot/llm/request'
MQTT_TOPIC_RESPONSE = 'robot/llm/response'
CLIENT_ID = f'fastapi-mqtt-server-{uuid.uuid4()}'

# Store awaiting responses
pending_requests = {}

# ---------- MQTT SETUP ----------
class MQTTHandler:
    def __init__(self):
        self.client = MQTTClient(CLIENT_ID)
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message

    def on_connect(self, client, flags, rc, properties):
        print(f"Connected to MQTT Broker at {MQTT_BROKER_HOST}")
        client.subscribe(MQTT_TOPIC_REQUEST)

    async def on_message(self, client, topic, payload, qos, properties):
        message = json.loads(payload)
        print(f"Received on {topic}: {message}")

        request_id = message.get("request_id")
        prompt = message.get("prompt")

        # Simulate LLM response (replace this with real LLM call)
        response_text = f"LLM Response to: {prompt}"

        response = {
            "request_id": request_id,
            "response": response_text
        }
        client.publish(MQTT_TOPIC_RESPONSE, json.dumps(response))

mqtt_handler = MQTTHandler()

# ---------- API MODEL ----------
class LLMRequest(BaseModel):
    prompt: str

# ---------- FASTAPI ENDPOINT ----------
@app.post("/query")
async def query_llm(req: LLMRequest):
    request_id = str(uuid.uuid4())
    payload = {
        "request_id": request_id,
        "prompt": req.prompt
    }
    pending_requests[request_id] = asyncio.Future()

    mqtt_handler.client.publish(MQTT_TOPIC_REQUEST, json.dumps(payload))

    try:
        result = await asyncio.wait_for(pending_requests[request_id], timeout=10)
        return {"response": result}
    except asyncio.TimeoutError:
        return {"error": "LLM response timeout"}
    finally:
        del pending_requests[request_id]

# ---------- MQTT RESPONSE HANDLER ----------
@mqtt_handler.client.on_message
async def handle_response(client, topic, payload, qos, properties):
    data = json.loads(payload)
    request_id = data.get("request_id")
    response = data.get("response")

    if request_id in pending_requests:
        pending_requests[request_id].set_result(response)

# ---------- MAIN ----------
if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.create_task(mqtt_handler.client.connect(MQTT_BROKER_HOST))
    uvicorn.run(app, host="0.0.0.0", port=8000)
    loop.run_forever()