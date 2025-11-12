"""Python SDK for Yale HPC jobs - simplified API similar to HuggingFace Jobs."""
import os
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, Union, List
import logging

from yale.cluster import ClusterConnection
from yale.jobs import YaleJob

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def read_manifest_list(file_path: str) -> List[str]:
    """Read IIIF manifest URLs from a text file.
    
    Args:
        file_path: Path to text file containing manifest URLs (one per line)
        
    Returns:
        List of manifest URLs
    """
    with open(file_path, 'r') as f:
        urls = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return urls


def _generate_preprocessing_script(raw_data_path: str, source_type: str, output_path: str, max_resolution: int = 2000) -> str:
    """Generate a preprocessing script to run on the cluster.
    
    Args:
        raw_data_path: Path to raw data on cluster (or URL for IIIF)
        source_type: Type of data source (pdf, directory, iiif, iiif-list)
        output_path: Path to save processed dataset
        max_resolution: Maximum image resolution in pixels
        
    Returns:
        Python script content
    """
    script = f"""# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "datasets",
#     "huggingface-hub",
#     "pillow",
#     "requests",
#     "pypdfium2",
# ]
# ///

import os
import sys
import logging
from pathlib import Path
from datasets import Dataset
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Starting data preprocessing on HPC cluster")
logger.info(f"Raw data path: {raw_data_path}")
logger.info(f"Output path: {output_path}")
logger.info(f"Source type: {source_type}")

# Add yale package to path if needed
sys.path.insert(0, str(Path(__file__).parent.parent))

# Process based on source type
"""
    
    if source_type == "pdf":
        script += f"""
# Import PDF processing
import pypdfium2 as pdfium

def pdf_to_images(pdf_path):
    \"\"\"Convert PDF to images.\"\"\"
    images = []
    pdf = pdfium.PdfDocument(pdf_path)
    
    for page_num in range(len(pdf)):
        page = pdf[page_num]
        pil_image = page.render(scale=2.0).to_pil()
        images.append(pil_image.convert("RGB"))
    
    return images

# Check if raw_data_path is a file or directory
raw_path = Path("{raw_data_path}")
all_images = []
metadata = []

if raw_path.is_file():
    logger.info(f"Processing single PDF: {{raw_path}}")
    images = pdf_to_images(str(raw_path))
    for i, img in enumerate(images):
        all_images.append(img)
        metadata.append({{
            "source": str(raw_path.name),
            "page": i + 1
        }})
else:
    logger.info(f"Processing PDF directory: {{raw_path}}")
    pdf_files = list(raw_path.glob("**/*.pdf"))
    logger.info(f"Found {{len(pdf_files)}} PDF files")
    
    for pdf_file in pdf_files:
        logger.info(f"Processing: {{pdf_file.name}}")
        images = pdf_to_images(str(pdf_file))
        for i, img in enumerate(images):
            all_images.append(img)
            metadata.append({{
                "source": str(pdf_file.name),
                "page": i + 1
            }})

# Create dataset
logger.info(f"Creating dataset with {{len(all_images)}} images")
dataset = Dataset.from_dict({{
    "image": all_images,
    "source": [m["source"] for m in metadata],
    "page": [m["page"] for m in metadata],
}})
"""
    elif source_type == "directory":
        script += f"""
# Process image directory
raw_path = Path("{raw_data_path}")
image_extensions = {{'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.gif'}}

logger.info(f"Processing image directory: {{raw_path}}")

# Find all image files
image_files = []
for ext in image_extensions:
    image_files.extend(raw_path.glob(f"**/*{{ext}}"))
    image_files.extend(raw_path.glob(f"**/*{{ext.upper()}}"))

logger.info(f"Found {{len(image_files)}} image files")

# Load images
all_images = []
metadata = []

for img_file in image_files:
    try:
        img = Image.open(img_file).convert("RGB")
        all_images.append(img)
        metadata.append({{
            "source": str(img_file.name),
            "path": str(img_file.relative_to(raw_path))
        }})
    except Exception as e:
        logger.warning(f"Failed to load {{img_file}}: {{e}}")

# Create dataset
logger.info(f"Creating dataset with {{len(all_images)}} images")
dataset = Dataset.from_dict({{
    "image": all_images,
    "source": [m["source"] for m in metadata],
    "path": [m["path"] for m in metadata],
}})
"""
    elif source_type == "iiif":
        script += f"""
# Process IIIF manifest
import json
import requests
from io import BytesIO

def load_iiif_manifest(url):
    \"\"\"Load IIIF manifest from URL.\"\"\"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

def detect_iiif_version(manifest):
    \"\"\"Detect IIIF manifest version.\"\"\"
    context = manifest.get("@context", "")
    if isinstance(context, str):
        if "iiif.io/api/presentation/3" in context:
            return "3"
        elif "iiif.io/api/presentation/2" in context:
            return "2"
    elif isinstance(context, list):
        for ctx in context:
            if isinstance(ctx, str):
                if "iiif.io/api/presentation/3" in ctx:
                    return "3"
                elif "iiif.io/api/presentation/2" in ctx:
                    return "2"
    if "items" in manifest:
        return "3"
    elif "sequences" in manifest:
        return "2"
    return "unknown"

def get_iiif_images(manifest, version):
    \"\"\"Extract images from IIIF manifest.\"\"\"
    images = []
    
    if version == "3":
        items = manifest.get("items", [])
        for canvas_idx, canvas in enumerate(items):
            for anno_page in canvas.get("items", []):
                for anno in anno_page.get("items", []):
                    body = anno.get("body", {{}})
                    if isinstance(body, dict):
                        url = body.get("id") or body.get("@id", "")
                        if "/info.json" in url:
                            url = url.replace("/info.json", "/full/max/0/default.jpg")
                        elif not any(ext in url.lower() for ext in [".jpg", ".jpeg", ".png"]):
                            if not url.endswith("/"):
                                url += "/"
                            url += "full/max/0/default.jpg"
                        
                        if url:
                            images.append({{"url": url, "canvas": canvas_idx + 1}})
    elif version == "2":
        sequences = manifest.get("sequences", [])
        for sequence in sequences:
            canvases = sequence.get("canvases", [])
            for canvas_idx, canvas in enumerate(canvases):
                for image_anno in canvas.get("images", []):
                    resource = image_anno.get("resource", {{}})
                    url = resource.get("id") or resource.get("@id", "")
                    if "/info.json" in url:
                        url = url.replace("/info.json", "/full/max/0/default.jpg")
                    elif not any(ext in url.lower() for ext in [".jpg", ".jpeg", ".png"]):
                        if not url.endswith("/"):
                            url += "/"
                        url += "full/max/0/default.jpg"
                    
                    if url:
                        images.append({{"url": url, "canvas": canvas_idx + 1}})
    
    return images

logger.info("Loading IIIF manifest from: {raw_data_path}")
manifest = load_iiif_manifest("{raw_data_path}")
version = detect_iiif_version(manifest)
logger.info(f"IIIF version: {{version}}")

image_infos = get_iiif_images(manifest, version)
logger.info(f"Found {{len(image_infos)}} images in manifest")

# Download and process images in batches to avoid OOM
BATCH_SIZE = 50
MAX_RESOLUTION = {max_resolution}

def resize_image(img, max_size={max_resolution}):
    \"\"\"Resize image to max dimension while maintaining aspect ratio.\"\"\"
    width, height = img.size
    if width <= max_size and height <= max_size:
        return img
    
    if width > height:
        new_width = max_size
        new_height = int((max_size / width) * height)
    else:
        new_height = max_size
        new_width = int((max_size / height) * width)
    
    return img.resize((new_width, new_height), Image.Resampling.LANCZOS)

all_datasets = []

for batch_start in range(0, len(image_infos), BATCH_SIZE):
    batch_end = min(batch_start + BATCH_SIZE, len(image_infos))
    logger.info(f"Processing batch {{batch_start//BATCH_SIZE + 1}}/{{(len(image_infos) + BATCH_SIZE - 1)//BATCH_SIZE}}")
    
    batch_images = []
    batch_metadata = []
    
    for i in range(batch_start, batch_end):
        info = image_infos[i]
        try:
            logger.info(f"  Downloading image {{i+1}}/{{len(image_infos)}}")
            
            # Modify IIIF URL to request smaller image if possible
            url = info["url"]
            if MAX_RESOLUTION < 2000 and "full/max/0/default" in url:
                # Request scaled version from IIIF server
                url = url.replace("full/max/0/default", f"full/!{{MAX_RESOLUTION}},{{MAX_RESOLUTION}}/0/default")
            
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content)).convert("RGB")
            
            # Resize if still too large
            img = resize_image(img, MAX_RESOLUTION)
            
            batch_images.append(img)
            batch_metadata.append({{"canvas": info["canvas"]}})
        except Exception as e:
            logger.warning(f"  Failed to download image {{i+1}}: {{e}}")
    
    # Create dataset for this batch
    if batch_images:
        batch_dataset = Dataset.from_dict({{
            "image": batch_images,
            "canvas": [m["canvas"] for m in batch_metadata],
        }})
        all_datasets.append(batch_dataset)
        logger.info(f"  Batch {{batch_start//BATCH_SIZE + 1}} complete: {{len(batch_images)}} images")

# Combine all batches
if all_datasets:
    from datasets import concatenate_datasets
    logger.info(f"Combining {{len(all_datasets)}} batches")
    dataset = concatenate_datasets(all_datasets)
    logger.info(f"Total images: {{len(dataset)}}")
else:
    logger.error("No images were downloaded")
    sys.exit(1)
"""
    elif source_type == "iiif-list":
        script += f"""
# Process IIIF manifest list
import json
import requests
from io import BytesIO
from datasets import concatenate_datasets

def load_iiif_manifest(url):
    \"\"\"Load IIIF manifest from URL.\"\"\"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

def detect_iiif_version(manifest):
    \"\"\"Detect IIIF manifest version.\"\"\"
    context = manifest.get("@context", "")
    if isinstance(context, str):
        if "iiif.io/api/presentation/3" in context:
            return "3"
        elif "iiif.io/api/presentation/2" in context:
            return "2"
    elif isinstance(context, list):
        for ctx in context:
            if isinstance(ctx, str):
                if "iiif.io/api/presentation/3" in ctx:
                    return "3"
                elif "iiif.io/api/presentation/2" in ctx:
                    return "2"
    if "items" in manifest:
        return "3"
    elif "sequences" in manifest:
        return "2"
    return "unknown"

def get_iiif_images(manifest, version):
    \"\"\"Extract images from IIIF manifest.\"\"\"
    images = []
    
    if version == "3":
        items = manifest.get("items", [])
        for canvas_idx, canvas in enumerate(items):
            for anno_page in canvas.get("items", []):
                for anno in anno_page.get("items", []):
                    body = anno.get("body", {{}})
                    if isinstance(body, dict):
                        url = body.get("id") or body.get("@id", "")
                        if "/info.json" in url:
                            url = url.replace("/info.json", "/full/max/0/default.jpg")
                        elif not any(ext in url.lower() for ext in [".jpg", ".jpeg", ".png"]):
                            if not url.endswith("/"):
                                url += "/"
                            url += "full/max/0/default.jpg"
                        
                        if url:
                            images.append({{"url": url, "canvas": canvas_idx + 1}})
    elif version == "2":
        sequences = manifest.get("sequences", [])
        for sequence in sequences:
            canvases = sequence.get("canvases", [])
            for canvas_idx, canvas in enumerate(canvases):
                for image_anno in canvas.get("images", []):
                    resource = image_anno.get("resource", {{}})
                    url = resource.get("id") or resource.get("@id", "")
                    if "/info.json" in url:
                        url = url.replace("/info.json", "/full/max/0/default.jpg")
                    elif not any(ext in url.lower() for ext in [".jpg", ".jpeg", ".png"]):
                        if not url.endswith("/"):
                            url += "/"
                        url += "full/max/0/default.jpg"
                    
                    if url:
                        images.append({{"url": url, "canvas": canvas_idx + 1}})
    
    return images

# Read manifest URLs from file
logger.info("Reading manifest list from: {raw_data_path}")
with open("{raw_data_path}", "r") as f:
    manifest_urls = [line.strip() for line in f if line.strip() and not line.startswith("#")]

logger.info(f"Found {{len(manifest_urls)}} manifest URLs")

# Download and process images in batches to avoid OOM
BATCH_SIZE = 50
MAX_RESOLUTION = {max_resolution}

def resize_image(img, max_size={max_resolution}):
    \"\"\"Resize image to max dimension while maintaining aspect ratio.\"\"\"
    width, height = img.size
    if width <= max_size and height <= max_size:
        return img
    
    if width > height:
        new_width = max_size
        new_height = int((max_size / width) * height)
    else:
        new_height = max_size
        new_width = int((max_size / height) * width)
    
    return img.resize((new_width, new_height), Image.Resampling.LANCZOS)

# Process each manifest
all_datasets = []
for manifest_idx, manifest_url in enumerate(manifest_urls, 1):
    logger.info(f"Processing manifest {{manifest_idx}}/{{len(manifest_urls)}}: {{manifest_url}}")
    
    try:
        manifest = load_iiif_manifest(manifest_url)
        version = detect_iiif_version(manifest)
        logger.info(f"  IIIF version: {{version}}")
        
        image_infos = get_iiif_images(manifest, version)
        logger.info(f"  Found {{len(image_infos)}} images")
        
        # Process images in batches
        manifest_datasets = []
        for batch_start in range(0, len(image_infos), BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, len(image_infos))
            logger.info(f"  Processing batch {{batch_start//BATCH_SIZE + 1}}/{{(len(image_infos) + BATCH_SIZE - 1)//BATCH_SIZE}}")
            
            batch_images = []
            batch_metadata = []
            
            for i in range(batch_start, batch_end):
                info = image_infos[i]
                try:
                    logger.info(f"    Downloading image {{i+1}}/{{len(image_infos)}}")
                    
                    # Modify IIIF URL to request smaller image if possible
                    url = info["url"]
                    if MAX_RESOLUTION < 2000 and "full/max/0/default" in url:
                        url = url.replace("full/max/0/default", f"full/!{{MAX_RESOLUTION}},{{MAX_RESOLUTION}}/0/default")
                    
                    response = requests.get(url, timeout=60)
                    response.raise_for_status()
                    img = Image.open(BytesIO(response.content)).convert("RGB")
                    
                    # Resize if still too large
                    img = resize_image(img, MAX_RESOLUTION)
                    
                    batch_images.append(img)
                    batch_metadata.append({{
                        "manifest_url": manifest_url,
                        "manifest_index": manifest_idx,
                        "canvas": info["canvas"]
                    }})
                except Exception as e:
                    logger.warning(f"    Failed to download image {{i+1}}: {{e}}")
            
            # Create dataset for this batch
            if batch_images:
                batch_dataset = Dataset.from_dict({{
                    "image": batch_images,
                    "manifest_url": [m["manifest_url"] for m in batch_metadata],
                    "manifest_index": [m["manifest_index"] for m in batch_metadata],
                    "canvas": [m["canvas"] for m in batch_metadata],
                }})
                manifest_datasets.append(batch_dataset)
                logger.info(f"    Batch complete: {{len(batch_images)}} images")
        
        # Combine batches for this manifest
        if manifest_datasets:
            if len(manifest_datasets) > 1:
                manifest_dataset = concatenate_datasets(manifest_datasets)
            else:
                manifest_dataset = manifest_datasets[0]
            all_datasets.append(manifest_dataset)
            logger.info(f"  Manifest complete: {{len(manifest_dataset)}} images total")
    except Exception as e:
        logger.error(f"  Failed to process manifest: {{e}}")

# Combine all datasets
if all_datasets:
    logger.info(f"Combining {{len(all_datasets)}} datasets")
    dataset = concatenate_datasets(all_datasets)
    logger.info(f"Combined dataset has {{len(dataset)}} images")
else:
    logger.error("No datasets were created")
    sys.exit(1)
"""
    
    script += f"""
# Save dataset
logger.info(f"Saving dataset to: {output_path}")
dataset.save_to_disk("{output_path}")
logger.info("✅ Preprocessing complete!")
"""
    
    return script


def run_job(
    script: str,
    data_source: Optional[Union[str, Any]] = None,
    source_type: str = "auto",
    job_name: str = "yale-job",
    gpus: str = "p100:2",
    partition: str = "gpu",
    cpus_per_task: int = 2,
    time_limit: str = "10:00",
    memory: Optional[str] = None,
    env: Optional[str] = None,
    hpc_process: bool = False,
    max_resolution: int = 2000,
    wait: bool = False,
    config_path: Optional[str] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
) -> YaleJob:
    """Run a job on Yale HPC cluster.
    
    Simple API similar to HuggingFace Jobs.
    
    Args:
        script: Python script content to run
        data_source: Data source (path, URL, dataset name, etc.)
        source_type: Type of data source (auto, pdf, iiif, web, directory, hf)
        job_name: Name for the job
        gpus: GPU specification (e.g., "p100:2", "v100:1")
        partition: SLURM partition
        cpus_per_task: Number of CPUs per task
        time_limit: Time limit (HH:MM:SS or HH:MM)
        memory: Memory limit (e.g., "32G")
        env: Conda environment name (overrides config.yaml)
        hpc_process: If True, process data on HPC (copy raw data first)
        wait: Whether to wait for job completion
        config_path: Path to config.yaml file
        username: Username for SSH connection
        password: Password for SSH connection
        
    Returns:
        YaleJob instance for job management
        
    Example:
        >>> from yale import run_job
        >>> 
        >>> script = '''
        ... import pandas as pd
        ... from datasets import load_from_disk
        ... 
        ... dataset = load_from_disk("dataset")
        ... print(f"Loaded {len(dataset)} samples")
        ... '''
        >>> 
        >>> job = run_job(
        ...     script=script,
        ...     data_source="path/to/pdfs",
        ...     job_name="my-ocr-job",
        ...     gpus="v100:2"
        ... )
    """
    import uuid
    run_id = str(uuid.uuid4())[:8]
    logger.info(f"[RUN_JOB CALLED] run_id={run_id}, job_name={job_name}")
    
    # Create cluster connection
    connection = ClusterConnection(config_path)
    
    # Override env in config if provided
    if env:
        connection.config['env'] = env
    
    connection.connect(username, password)
    
    try:
        # Create job manager
        job = YaleJob(connection, job_name=job_name)
        logger.info(f"[RUN_JOB {run_id}] Created YaleJob instance")
        
        # Prepare data if provided
        dataset_path = None
        raw_data_path = None
        if data_source:
            if hpc_process:
                # For HPC processing, we always get raw data path and need preprocessing
                result = job.prepare_data(data_source, source_type, hpc_process=True)
                detected_type = source_type if source_type != "auto" else job._detect_source_type(data_source)
                
                # All HPC processing requires preprocessing script
                if detected_type in ["pdf", "directory", "iiif", "iiif-list"]:
                    raw_data_path = result
                else:
                    # For other types (e.g., HF datasets), use directly
                    dataset_path = result
            else:
                dataset_path = job.prepare_data(data_source, source_type, hpc_process=False)
        
        # Generate preprocessing script if we have raw data
        if raw_data_path:
            detected_type = source_type if source_type != "auto" else job._detect_source_type(data_source)
            
            # Set dataset_path to the output of preprocessing
            dataset_path = f"{job.job_dir}/{job_name}_data"
            
            # Pass max_resolution to preprocessing script
            preprocess_script = _generate_preprocessing_script(
                raw_data_path, 
                detected_type,
                dataset_path,
                max_resolution
            )
            
            # Save and upload preprocessing script
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(preprocess_script)
                local_preprocess_script = f.name
            
            remote_preprocess_script = f"{job.job_dir}/{job_name}_preprocess.py"
            connection.upload_file(local_preprocess_script, remote_preprocess_script)
        
        # Modify script to use the dataset if provided
        if dataset_path and "load_from_disk" in script:
            # Replace placeholder with actual path
            script = script.replace('load_from_disk("dataset")', f'load_from_disk("{dataset_path}")')
        
        # Save Python script to temporary file
        logger.info(f"[RUN_JOB {run_id}] Saving Python script to temp file...")
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(script)
            local_python_script = f.name
        logger.info(f"[RUN_JOB {run_id}] Saved to: {local_python_script}")
        
        # Upload Python script to cluster
        remote_python_script = f"{job.job_dir}/{job_name}.py"
        logger.info(f"[RUN_JOB {run_id}] Uploading Python script to: {remote_python_script}")
        connection.upload_file(local_python_script, remote_python_script)
        logger.info(f"[RUN_JOB {run_id}] Python script uploaded successfully")
        
        # Create SLURM batch script that calls the Python script
        # If we have preprocessing, run it first
        if raw_data_path:
            bash_command = f"uv run {remote_preprocess_script} && uv run {remote_python_script}"
        else:
            bash_command = f"uv run {remote_python_script}"
        sbatch_content = job.create_sbatch_script(
            script_content=bash_command,
            cpus_per_task=cpus_per_task,
            gpus=gpus,
            partition=partition,
            time_limit=time_limit,
            memory=memory,
        )
        
        # Save batch script to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
            f.write(sbatch_content)
            local_sbatch_script = f.name
        
        # Upload batch script to cluster
        remote_sbatch_script = f"{job.job_dir}/{job_name}.sh"
        connection.upload_file(local_sbatch_script, remote_sbatch_script)
        
        # Submit job
        logger.info(f"[RUN_JOB {run_id}] About to call job.submit()")
        job.submit(remote_sbatch_script, wait=wait)
        logger.info(f"[RUN_JOB {run_id}] job.submit() completed")
        
        # Clean up temporary files
        try:
            if raw_data_path and 'local_preprocess_script' in locals():
                os.unlink(local_preprocess_script)
            if 'local_python_script' in locals():
                os.unlink(local_python_script)
            if 'local_sbatch_script' in locals():
                os.unlink(local_sbatch_script)
        except Exception as cleanup_error:
            logger.warning(f"Failed to clean up temp files: {cleanup_error}")
        
        return job
        
    except Exception as e:
        connection.close()
        raise e


def run_ocr_job(
    data_source: Union[str, Any],
    output_dataset: str,
    source_type: str = "auto",
    model: str = "rednote-hilab/dots.ocr",
    ocr_engine: str = "dots-ocr",
    batch_size: int = 16,
    max_samples: Optional[int] = None,
    job_name: str = "yale-job",
    gpus: str = "p100:2",
    partition: str = "gpu",
    time_limit: str = "02:00:00",
    env: Optional[str] = None,
    prompt_mode: str = "layout-all",
    output_column: str = "text",
    dataset_path: Optional[str] = None,
    max_model_len: int = 32768,
    max_tokens: int = 16384,
    max_resolution: int = 2000,
    hpc_process: bool = False,
    wait: bool = False,
    config_path: Optional[str] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
) -> YaleJob:
    """Run an OCR job on Yale HPC cluster.
    
    Convenience function for running OCR on documents/images.
    
    Args:
        data_source: Data source (PDF, IIIF, directory, etc.)
        output_dataset: Name for output dataset
        source_type: Type of data source (auto, pdf, iiif, web, directory, hf)
        model: OCR model to use
        ocr_engine: OCR engine/script to use (dots-ocr, nanonets2-ocr, deepseek-ocr, qwen3-vl; default: dots-ocr)
        batch_size: Batch size for processing
        max_samples: Maximum number of samples to process
        job_name: Name for the job
        gpus: GPU specification
        partition: SLURM partition (default: gpu)
        time_limit: Time limit in HH:MM:SS format (default: 02:00:00)
        env: Conda environment name (overrides config.yaml)
        prompt_mode: DoTS.ocr prompt mode (ocr, layout-all, layout-only; default: layout-all) or custom prompt for vision models
        output_column: Output column name for results (default: text)
        dataset_path: Path to existing dataset on cluster (skips data upload if provided)
        max_model_len: Maximum model context length (default: 32768)
        max_tokens: Maximum output tokens (default: 16384)
        max_resolution: Maximum image resolution in pixels (default: 2000)
        hpc_process: If True, process data on HPC (copy raw data first)
        wait: Whether to wait for job completion
        config_path: Path to config.yaml file
        username: Username for SSH connection
        password: Password for SSH connection
        
    Returns:
        YaleJob instance for job management
        
    Example:
        >>> from yale import run_ocr_job
        >>> 
        >>> job = run_ocr_job(
        ...     data_source="manuscripts/",
        ...     output_dataset="manuscripts-ocr",
        ...     source_type="pdf",
        ...     gpus="v100:2"
        ... )
        >>> 
        >>> # Check status
        >>> status = job.get_status()
        >>> print(status)
    """
    logger.info(f"[RUN_OCR_JOB CALLED] job_name={job_name}, source={data_source}, ocr_engine={ocr_engine}")
    
    # Map OCR engine to script file
    ocr_script_map = {
        "dots-ocr": "yale/ocr/yale-dots-ocr.py",
        "nanonets2-ocr": "yale/ocr/nanonets2-ocr.py",
        "deepseek-ocr": "yale/ocr/deepseek-ocr.py",
        "qwen3-vl": "yale/ocr/qwen3-vl.py",
    }
    
    if ocr_engine not in ocr_script_map:
        raise ValueError(f"Unknown OCR engine: {ocr_engine}. Choose from: {list(ocr_script_map.keys())}")
    
    # Get path to OCR script
    ocr_script_path = Path(__file__).parent / ".." / ocr_script_map[ocr_engine]
    ocr_script_path = ocr_script_path.resolve()
    
    if not ocr_script_path.exists():
        raise FileNotFoundError(f"OCR script not found: {ocr_script_path}")
    
    # Read the OCR script
    logger.info(f"Using OCR script: {ocr_script_path}")
    with open(ocr_script_path, 'r') as f:
        ocr_script_template = f.read()
    
    # Extract the main() function and dependencies from the OCR script
    # Remove the if __name__ == "__main__" block
    script_lines = ocr_script_template.split('\n')
    main_block_start = None
    for i, line in enumerate(script_lines):
        if 'if __name__ == "__main__":' in line:
            main_block_start = i
            break
    
    if main_block_start:
        # Keep everything before the main block
        ocr_core_script = '\n'.join(script_lines[:main_block_start])
    else:
        ocr_core_script = ocr_script_template
    
    # Create wrapper script that calls main() directly
    prompt_mode_arg = f", prompt_mode='{prompt_mode}'" if ocr_engine == "dots-ocr" else ""
    custom_prompt_arg = f", custom_prompt='{prompt_mode}'" if ocr_engine == "qwen3-vl" and prompt_mode else ""
    output_column_arg = f", output_column='{output_column}'" if output_column != "text" else ""
    max_samples_arg = f", max_samples={max_samples}" if max_samples else ""
    
    # Use placeholder that will be replaced by run_job() with the absolute path
    # run_job() will replace this with the full dataset path
    ocr_script = ocr_core_script + f"""

# Yale Jobs SDK Integration - Direct main() call
if __name__ == "__main__":
    import os
    # Get the script's directory to build absolute path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(script_dir, "{job_name}_data")
    
    main(
        input_dataset=dataset_path,
        output_dataset="{output_dataset}",
        load_from_disk_path=True,
        batch_size={batch_size},
        max_model_len={max_model_len},
        max_tokens={max_tokens},
        model="{model}"{prompt_mode_arg}{custom_prompt_arg}{output_column_arg}{max_samples_arg}
    )
"""
    
    # If dataset_path is provided, skip data upload and use the existing path
    if dataset_path:
        logger.info(f"Using existing dataset at: {dataset_path}")
        # Replace the dynamic path construction with the provided path
        ocr_script = ocr_script.replace(
            f'dataset_path = os.path.join(script_dir, "{job_name}_data")',
            f'dataset_path = "{dataset_path}"'
        )
        # Set data_source to None to skip upload
        actual_data_source = None
    else:
        actual_data_source = data_source
    
    return run_job(
        script=ocr_script,
        data_source=actual_data_source,
        source_type=source_type,
        job_name=job_name,
        gpus=gpus,
        partition=partition,
        time_limit=time_limit,
        env=env,
        hpc_process=hpc_process,
        max_resolution=max_resolution,
        wait=wait,
        config_path=config_path,
        username=username,
        password=password,
    )


class YaleJobs:
    """Yale Jobs SDK - object-oriented API."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize Yale Jobs SDK.
        
        Args:
            config_path: Path to config.yaml file
        """
        self.config_path = config_path
        self.connection = None
    
    def connect(self, username: Optional[str] = None, password: Optional[str] = None):
        """Connect to the cluster.
        
        Args:
            username: Username for SSH connection
            password: Password for SSH connection
        """
        self.connection = ClusterConnection(self.config_path)
        self.connection.connect(username, password)
        logger.info("Connected to Yale HPC")
    
    def submit_job(
        self,
        script: str,
        data_source: Optional[Union[str, Any]] = None,
        source_type: str = "auto",
        job_name: str = "yale-job",
        **kwargs
    ) -> YaleJob:
        """Submit a job to the cluster.
        
        Args:
            script: Python script content to run
            data_source: Data source (path, URL, dataset name, etc.)
            source_type: Type of data source
            job_name: Name for the job
            **kwargs: Additional job parameters (gpus, partition, etc.)
            
        Returns:
            YaleJob instance
        """
        if not self.connection:
            raise ConnectionError("Not connected. Call connect() first.")
        
        job = YaleJob(self.connection, job_name=job_name)
        
        # Prepare data if provided
        if data_source:
            dataset_path = job.prepare_data(data_source, source_type)
            if "load_from_disk" in script:
                script = script.replace('load_from_disk("dataset")', f'load_from_disk("{dataset_path}")')
        
        # Save Python script
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(script)
            local_python_script = f.name
        
        remote_python_script = f"{job.job_dir}/{job_name}.py"
        self.connection.upload_file(local_python_script, remote_python_script)
        
        # Create and upload batch script that calls Python script
        bash_command = f"python {remote_python_script}"
        sbatch_content = job.create_sbatch_script(bash_command, **kwargs)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
            f.write(sbatch_content)
            local_sbatch_script = f.name
        
        remote_sbatch_script = f"{job.job_dir}/{job_name}.sh"
        self.connection.upload_file(local_sbatch_script, remote_sbatch_script)
        
        # Submit job
        job.submit(remote_sbatch_script)
        
        return job
    
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get status of a job.
        
        Args:
            job_id: Job ID
            
        Returns:
            Job status dict
        """
        if not self.connection:
            raise ConnectionError("Not connected. Call connect() first.")
        
        job = YaleJob(self.connection)
        job.job_id = job_id
        return job.get_status()
    
    def close(self):
        """Close the connection."""
        if self.connection:
            self.connection.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

