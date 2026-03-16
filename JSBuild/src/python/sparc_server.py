# sparc_server.py

import io
import torch
from flask import Flask, request, send_file, jsonify
from sparc3d_sdf.scripts.sdf import run

app = Flask(__name__)

MODEL_ID = "NewBie-AI/NewBie-image-Exp0.1"

pipe = None
DEVICE = "cpu"
DTYPE = torch.float32


# --------------------------------------------------
# Hardware Detection
# --------------------------------------------------

def detect_device():
    global DEVICE

    if torch.cuda.is_available():
        DEVICE = "cuda"

        gpu_name = torch.cuda.get_device_name(0).lower()

        if "amd" in gpu_name or "radeon" in gpu_name:
            print("Detected ROCm GPU:", gpu_name)
        else:
            print("Detected CUDA GPU:", gpu_name)

    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        DEVICE = "mps"
        print("Detected Apple Silicon GPU (MPS)")

    else:
        DEVICE = "cpu"
        print("Falling back to CPU")

    return DEVICE


def get_dtype(device):

    if device == "cuda":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16

    if device == "mps":
        return torch.float16

    return torch.float32


# --------------------------------------------------
# Pipeline Loader
# --------------------------------------------------

def load_pipeline():
    global pipe
    global DEVICE
    global DTYPE

    DEVICE = detect_device()
    DTYPE = get_dtype(DEVICE)

    print("AI Runtime")
    print("-----------")
    print("Device:", DEVICE)
    print("DType:", DTYPE)
    print("Model:", MODEL_ID)

    pipe = NewbiePipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE
    )

    pipe = pipe.to(DEVICE)

    # Diffusers performance tweaks
    if DEVICE == "cuda":
        pipe.enable_attention_slicing()
        pipe.enable_xformers_memory_efficient_attention()


load_pipeline()


# --------------------------------------------------
# Image Generation Endpoint
# --------------------------------------------------

@app.route("/generate_image", methods=["POST"])
def generate_image():

    data = request.json

    prompt = data["prompt"]
    height = data.get("height", 1024)
    width = data.get("width", 1024)
    steps = data.get("inference_steps", 30)

    result = pipe(
        prompt,
        height=height,
        width=width,
        num_inference_steps=steps
    )

    image = result.images[0]

    img_bytes = io.BytesIO()
    image.save(img_bytes, format="PNG")
    img_bytes.seek(0)

    return send_file(img_bytes, mimetype="image/png")


# --------------------------------------------------
# SDF Generation Endpoint
# --------------------------------------------------

@app.route("/generate_sdf", methods=["POST"])
def generate_sdf():

    data = request.json

    prompt = data["prompt"]
    user_id = data["userID"]
    n = data.get("n", 1024)

    input_obj = f"assets/{user_id}.obj"
    output_obj = f"./local_generations/{user_id}_{n}.obj"

    sdf_file = run(input_obj, n, prompt, output_obj)

    return send_file(sdf_file)


# --------------------------------------------------
# Runtime Info Endpoint (useful for debugging)
# --------------------------------------------------

@app.route("/runtime", methods=["GET"])
def runtime_info():

    info = {
        "device": DEVICE,
        "dtype": str(DTYPE),
        "cuda_available": torch.cuda.is_available(),
        "mps_available": hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    }

    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)

    return jsonify(info)


# --------------------------------------------------
# Start Server
# --------------------------------------------------

if __name__ == "__main__":
    app.run(port=5001)