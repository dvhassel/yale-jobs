# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "datasets",
#     "huggingface-hub[hf_transfer]",
#     "pillow",
#     "vllm>=0.6.1",
#     "tqdm",
#     "toolz",
#     "torch",
# ]
#
# ///

"""
Yale-adapted Qwen3-VL script for use with Yale HPC Jobs.

This version is designed to work with Yale's job submission system and
supports all Yale data sources (PDF, IIIF, directories, web URLs, HF datasets).

Features:
- Flexible prompt customization
- High-quality text transcription
- Vision-language understanding
- Supports various image formats

Usage with Yale Jobs:
    yale jobs ocr path/to/pdfs output-dataset --source-type pdf --gpus v100:2
    yale jobs ocr https://example.com/manifest.json output --source-type iiif
"""

import argparse
import base64
import io
import json
import logging
import os
import sys
from typing import Any, Dict, List, Union
from datetime import datetime

import torch
from datasets import load_dataset, load_from_disk, Dataset
from PIL import Image
from toolz import partition_all
from tqdm.auto import tqdm
from vllm import LLM, SamplingParams

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_cuda_availability():
    """Check if CUDA is available and exit if not."""
    if not torch.cuda.is_available():
        logger.error("CUDA is not available. This script requires a GPU.")
        logger.error("Please run on a machine with a CUDA-capable GPU.")
        sys.exit(1)
    else:
        logger.info(f"CUDA is available. GPU: {torch.cuda.get_device_name(0)}")


def make_vision_message(
    image: Union[Image.Image, Dict[str, Any], str],
    prompt: str = "Transcribe this image to text. Just return the text",
) -> List[Dict]:
    """Create chat message for vision processing."""
    # Convert to PIL Image if needed
    if isinstance(image, Image.Image):
        pil_img = image
    elif isinstance(image, dict) and "bytes" in image:
        pil_img = Image.open(io.BytesIO(image["bytes"]))
    elif isinstance(image, str):
        pil_img = Image.open(image)
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")

    # Convert to RGB
    pil_img = pil_img.convert("RGB")

    # Convert to base64 data URI
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    data_uri = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"

    # Return message in vLLM format
    return [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_uri}},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def main(
    input_dataset: str,
    output_dataset: str,
    image_column: str = "image",
    batch_size: int = 16,
    model: str = "Qwen/Qwen2-VL-7B-Instruct",
    max_model_len: int = 32768,
    max_tokens: int = 8192,
    gpu_memory_utilization: float = 0.8,
    split: str = "train",
    max_samples: int = None,
    shuffle: bool = False,
    seed: int = 42,
    output_column: str = "text",
    load_from_disk_path: bool = False,
    custom_prompt: str = None,
):
    """Process images from dataset through Qwen3-VL model.
    
    Args:
        input_dataset: Input dataset ID or path
        output_dataset: Output dataset path
        image_column: Column containing images
        batch_size: Batch size for processing
        model: Model to use
        max_model_len: Maximum model context length
        max_tokens: Maximum tokens to generate
        gpu_memory_utilization: GPU memory utilization
        split: Dataset split
        max_samples: Maximum samples to process
        shuffle: Whether to shuffle dataset
        seed: Random seed
        output_column: Output column name
        load_from_disk_path: Whether input is a disk path (for Yale jobs)
        custom_prompt: Custom prompt for vision processing
    """

    # Set cache directories to avoid disk quota issues
    # # Use current directory (project space) instead of home directory
    # script_dir = os.path.dirname(os.path.abspath(__file__))
    # cache_dir = os.path.join(script_dir, ".cache")
    # os.makedirs(cache_dir, exist_ok=True)
    
    # # Redirect vLLM and torch caches to project directory
    # os.environ["VLLM_CACHE_DIR"] = cache_dir
    # os.environ["XDG_CACHE_HOME"] = cache_dir
    # os.environ["TORCH_HOME"] = os.path.join(cache_dir, "torch")
    # os.environ["HF_HOME"] = os.path.join(cache_dir, "huggingface")
    
    # logger.info(f"Using cache directory: {cache_dir}")

    # Check CUDA availability first
    check_cuda_availability()

    # Track processing start time
    start_time = datetime.now()

    # Set prompt
    prompt = custom_prompt if custom_prompt else "Transcribe this image to text. Just return the text"
    logger.info(f"Using prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")

    # Load dataset
    logger.info(f"Loading dataset: {input_dataset}")
    
    if load_from_disk_path:
        # Load from disk (for Yale jobs with prepared data)
        dataset = load_from_disk(input_dataset)
    else:
        # Load from HuggingFace Hub
        dataset = load_dataset(input_dataset, split=split)

    # Validate image column
    if image_column not in dataset.column_names:
        raise ValueError(
            f"Column '{image_column}' not found. Available: {dataset.column_names}"
        )

    # Shuffle if requested
    if shuffle:
        logger.info(f"Shuffling dataset with seed {seed}")
        dataset = dataset.shuffle(seed=seed)

    # Limit samples if requested
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
        logger.info(f"Limited to {len(dataset)} samples")

    # Initialize vLLM
    logger.info(f"Initializing vLLM with model: {model}")
    llm = LLM(
        model=model,
        trust_remote_code=True,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        limit_mm_per_prompt={"image": 1},
    )

    sampling_params = SamplingParams(
        temperature=0.0,  # Deterministic for transcription
        max_tokens=max_tokens,
    )

    logger.info(f"Processing {len(dataset)} images in batches of {batch_size}")
    logger.info(f"Output will be written to column: {output_column}")

    # Process images in batches
    all_outputs = []

    for batch_indices in tqdm(
        partition_all(batch_size, range(len(dataset))),
        total=(len(dataset) + batch_size - 1) // batch_size,
        desc="Qwen3-VL processing",
    ):
        batch_indices = list(batch_indices)
        batch_images = [dataset[i][image_column] for i in batch_indices]

        try:
            # Create messages for batch
            batch_messages = [make_vision_message(img, prompt) for img in batch_images]

            # Process with vLLM
            outputs = llm.chat(batch_messages, sampling_params)

            # Extract outputs
            for output in outputs:
                text = output.outputs[0].text.strip()
                all_outputs.append(text)

        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            # Add error placeholders for failed batch
            all_outputs.extend(["[VISION ERROR]"] * len(batch_images))

    # Calculate processing time
    processing_duration = datetime.now() - start_time
    processing_time_str = f"{processing_duration.total_seconds() / 60:.1f} min"

    # Add output column to dataset
    logger.info(f"Adding '{output_column}' column to dataset")
    dataset = dataset.add_column(output_column, all_outputs)

    # Handle inference_info tracking (for multi-model comparisons)
    inference_entry = {
        "model_id": model,
        "column_name": output_column,
        "timestamp": datetime.now().isoformat(),
        "prompt": prompt,
    }

    if "inference_info" in dataset.column_names:
        # Append to existing inference info
        logger.info("Updating existing inference_info column")

        def update_inference_info(example):
            try:
                existing_info = json.loads(example["inference_info"]) if example["inference_info"] else []
            except (json.JSONDecodeError, TypeError):
                existing_info = []

            existing_info.append(inference_entry)
            return {"inference_info": json.dumps(existing_info)}

        dataset = dataset.map(update_inference_info)
    else:
        # Create new inference_info column
        logger.info("Creating new inference_info column")
        inference_list = [json.dumps([inference_entry])] * len(dataset)
        dataset = dataset.add_column("inference_info", inference_list)

    # Save to disk
    logger.info(f"Saving to {output_dataset}")
    dataset.save_to_disk(output_dataset)

    logger.info("✅ Qwen3-VL processing complete!")
    logger.info(f"Dataset saved to: {output_dataset}")
    logger.info(f"Processing time: {processing_time_str}")


if __name__ == "__main__":
    # Show example usage if no arguments
    if len(sys.argv) == 1:
        print("=" * 80)
        print("Yale Qwen3-VL Vision Document Processing")
        print("=" * 80)
        print("\nPowerful vision-language model for Yale HPC")
        print("\nFeatures:")
        print("- 🔤 High-quality text transcription")
        print("- 🎯 Custom prompt support")
        print("- 🖼️ Vision-language understanding")
        print("- 🌍 Multilingual support")
        print("\nUsage with Yale Jobs CLI:")
        print("\n1. OCR on PDFs:")
        print("   yale jobs ocr path/to/pdfs output-dataset --source-type pdf")
        print("\n2. OCR on IIIF manifest:")
        print("   yale jobs ocr https://example.com/manifest.json output --source-type iiif")
        print("\n3. OCR on image directory:")
        print("   yale jobs ocr path/to/images output --source-type directory")
        print("\n4. Direct script usage (on cluster):")
        print("   python qwen3-vl.py dataset_path output_path --load-from-disk-path")
        print("\n5. With custom prompt:")
        print('   python qwen3-vl.py dataset output --custom-prompt "Extract all text"')
        print("\n" + "=" * 80)
        print("\nFor full help, run: python qwen3-vl.py --help")
        sys.exit(0)

    parser = argparse.ArgumentParser(
        description="Document OCR using Qwen3-VL for Yale HPC",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("input_dataset", help="Input dataset path or ID")
    parser.add_argument("output_dataset", help="Output dataset path")
    parser.add_argument(
        "--image-column",
        default="image",
        help="Column containing images (default: image)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for processing (default: 16)",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2-VL-7B-Instruct",
        help="Model to use (default: Qwen/Qwen2-VL-7B-Instruct)",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=32768,
        help="Maximum model context length (default: 32768)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=8192,
        help="Maximum tokens to generate (default: 8192)",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.8,
        help="GPU memory utilization (default: 0.8)",
    )
    parser.add_argument(
        "--split", default="train", help="Dataset split to use (default: train)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Maximum number of samples to process",
    )
    parser.add_argument(
        "--shuffle", action="store_true", help="Shuffle dataset before processing"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling (default: 42)",
    )
    parser.add_argument(
        "--output-column",
        default="text",
        help="Column name for output text (default: text)",
    )
    parser.add_argument(
        "--load-from-disk-path",
        action="store_true",
        help="Input is a disk path (for Yale jobs prepared data)",
    )
    parser.add_argument(
        "--custom-prompt",
        help='Custom prompt for vision processing (default: "Transcribe this image to text. Just return the text")',
    )

    args = parser.parse_args()

    main(
        input_dataset=args.input_dataset,
        output_dataset=args.output_dataset,
        image_column=args.image_column,
        batch_size=args.batch_size,
        model=args.model,
        max_model_len=args.max_model_len,
        max_tokens=args.max_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        split=args.split,
        max_samples=args.max_samples,
        shuffle=args.shuffle,
        seed=args.seed,
        output_column=args.output_column,
        load_from_disk_path=args.load_from_disk_path,
        custom_prompt=args.custom_prompt,
    )

