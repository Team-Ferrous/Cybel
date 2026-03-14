import os
from flask         import Flask, request, jsonify
from hf_modelscope import ModelHub  # your class

app = Flask(__name__)
hub = ModelHub()

@app.route("/installed_models", methods=["GET"])
def installed_models():
    models = []
    local_dir = "./local_models"
    if os.path.exists(local_dir):
        models = [name for name in os.listdir(local_dir) if os.path.isdir(os.path.join(local_dir, name))]
    return jsonify({"models": models})

@app.route("/download_model", methods=["POST"])
def download_model():
    data = request.json
    model_name = data.get("modelName")
    sources  = data.get("sources", [])
    hf_token = data.get("hfToken")
    ms_token = data.get("msToken")

    try:
        if "hf" in sources:
            hub.hfToken = hf_token
            # You could save locally via hf_hub_download
            hub.download_huggingface_model(model_name)

        if "ms" in sources:
            hub.msToken = ms_token
            hub.download_modelscope_model(model_name)
            
        #Verso uses Ollama and thus doesn't need its own source function
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})