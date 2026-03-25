import torch
from diffusers import NewbiePipeline

MODEL_ID = "NewBie-AI/NewBie-image-Exp0.1"


def detect_device():
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0).lower()

        if "amd" in gpu_name or "radeon" in gpu_name:
            print("Detected ROCm GPU:", gpu_name)
        else:
            print("Detected CUDA GPU:", gpu_name)

    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        print("Detected Apple Silicon GPU (MPS)")

    else:
        device = "cpu"
        print("Falling back to CPU")

    return device


def choose_dtype(device):
    if device == "cuda":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16

    if device == "mps":
        return torch.float16

    return torch.float32


def run_newbie_pipeline(prompt):

    device = detect_device()
    dtype = choose_dtype(device)

    print("\nAI Runtime")
    print("-----------")
    print("Device:", device)
    print("DType:", dtype)
    print("Model:", MODEL_ID)
    print()

    pipe = NewbiePipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
    )

    pipe = pipe.to(device)

    # Optional performance tweaks
    if device == "cuda":
        pipe.enable_attention_slicing()

    #prompt = "1girl"

    result = pipe(
        prompt,
        height=1024,
        width=1024,
        num_inference_steps=28,
    )

    image = result.images[0]

    image.save("newbie_sample.png")

    print("Saved to newbie_sample.png")


if __name__ == "__main__":
    main()