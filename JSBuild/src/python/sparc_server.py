# sparc_server.py

import torch
from flask import Flask, request, send_file, jsonify
#from sparc3d_sdf.scripts.sdf import run

app = Flask(__name__)

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

def generate_sdf_local(user_id, prompt, n):
    input_obj = f"assets/{user_id}.obj"
    output_obj = f"./local_generations/{user_id}_{n}.obj"
    #run(input_obj, n, prompt, output_obj)
    print(f"SDF Built at Output Path: {output_obj}")

# WEB API
# --------------------------------------------------
# SDF Generation Endpoint
# --------------------------------------------------
@app.route("/generate_sdf", methods=["POST"])
def generate_sdf():

    data = request.json
    prompt  = data["prompt"]
    user_id = data["userID"]
    n       = data.get("n", 1024)

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
        "dtype": str(get_dtype(DEVICE)),
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