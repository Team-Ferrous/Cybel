# sparc_server.py
import torch
from flask     import Flask, request, send_file
from diffusers import NewbiePipeline
from sparc3d_sdf.scripts.sdf import run  # ✅ correct local import

app = Flask(__name__)

@app.route("/generate_image", methods=["POST"])
def generate_image():
    prompt = request.json["prompt"]
    model_id = "NewBie-AI/NewBie-image-Exp0.1"

    # Load pipeline
    pipe = NewbiePipeline.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
    ).to("cuda")
  # use float16 if your GPU does not support bfloat16
    image = pipe(
        prompt.query,
        height=prompt.height,
        width=prompt.width,
        num_inference_steps=prompt.inference_steps,
    ).images[0]

    image.save("newbie_sample.png")
    print("Saved to newbie_sample.png")
    return send_file(image)


@app.route("/generate_sdf", methods=["POST"])
def generate():
    prompt = request.json["prompt"]
    sdf_file = run("assets/{prompt.userID}.obj", prompt.n, prompt, f"./local_generations/{prompt.userID}_{prompt.n}.obj") #This is what run expects: -i assets/plane.obj --N 1024 -o plane_1024.obj
    return send_file(sdf_file)

app.run(port=5001)