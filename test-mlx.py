import inspect
from pprint import pprint

import mlx.core as mx
from mlx_vlm import load
from mlx_vlm.generate import batch_generate

# Load model
model_path = "Qwen/Qwen3-VL-4B-Instruct"
# model_path = "HuggingFaceTB/SmolVLM-500M-Instruct"

model, processor = load(model_path)
config = model.config

# Use the same image twice for testing
images = [
    "/Users/wjm55/Downloads/1994.A.0112_001_003_0010.jpg",
    "/Users/wjm55/Downloads/1994.A.0112_001_003_0010.jpg"
]

# Prompts for both images
prompts = [
    "Describe the image in detail.",
    "What objects or elements are visible in the image?"
]

# Generate output
result = batch_generate(
    model,
    processor=processor,
    images=images,
    prompts=prompts,
    max_tokens=500,
    verbose=True,
    resize_shape=(800, 800)
)

# Display results nicely
for i, text in enumerate(result.texts):
    print(f"—— Response {i+1} ——")
    print("Query:", prompts[i])
    print("Response:", text)
    print()
