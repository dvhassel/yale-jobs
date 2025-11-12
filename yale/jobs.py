"""Yale HPC job submission and management."""
import os
import time
import json
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
from datetime import datetime
import logging

from yale.cluster import ClusterConnection
from yale.data import (
    PDFDataSource, IIIFDataSource, WebDataSource,
    DirectoryDataSource, HFDataSource
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class YaleJob:
    """Manage SLURM jobs on Yale HPC."""
    
    def __init__(
        self,
        connection: ClusterConnection,
        job_name: str = "yale-job",
        job_dir: Optional[str] = None,
        result_dir: Optional[str] = None,
    ):
        """Initialize Yale job manager.
        
        Args:
            connection: Active cluster connection
            job_name: Name for the job
            job_dir: Directory on cluster for job files (from config if not specified)
            result_dir: Directory on cluster for results (from config if not specified)
        """
        self.connection = connection
        self.job_name = job_name
        
        # Use config values if not specified
        config = connection.config
        self.job_dir = job_dir or config.get('job_dir', 'yale_jobs')
        self.result_dir = result_dir or config.get('result_dir', 'yale_jobs/results')
        
        # Expand wildcards in paths
        self.job_dir = self._expand_path(self.job_dir)
        self.result_dir = self._expand_path(self.result_dir)
        
        self.job_id = None
        
    def _expand_path(self, path: str) -> str:
        """Expand wildcards in path using cluster commands.
        
        Args:
            path: Path potentially containing wildcards like proj*
            
        Returns:
            Expanded path
        """
        if '*' in path:
            result = self.connection.execute_command(f"echo {path}")
            expanded = result['stdout'].strip()
            if expanded and not expanded.startswith('echo'):
                return expanded
        return path
    
    def create_sbatch_script(
        self,
        script_content: str,
        cpus_per_task: int = 2,
        gpus: str = "p100:2",
        partition: str = "gpu",
        time_limit: str = "10:00",
        memory: Optional[str] = None,
        output_file: Optional[str] = None,
    ) -> str:
        """Create SLURM batch script.
        
        Args:
            script_content: Python/bash commands to run
            cpus_per_task: Number of CPUs per task
            gpus: GPU specification (e.g., "p100:2")
            partition: SLURM partition
            time_limit: Time limit (HH:MM:SS or HH:MM)
            memory: Memory limit (e.g., "32G")
            output_file: Output log file name
            
        Returns:
            SLURM batch script as string
        """
        # Use full path for output file in job_dir
        if output_file is None:
            output_file = f"{self.job_dir}/{self.job_name}.out"
        
        script = f"""#!/bin/bash

#SBATCH --job-name={self.job_name}
#SBATCH --output={output_file}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --gpus={gpus}
#SBATCH --partition={partition}
#SBATCH --time={time_limit}
"""
        
        if memory:
            script += f"#SBATCH --mem={memory}\n"
        
        script += "\n"
        
        # Install uv if not already available
        script += "# Install uv for dependency management\n"
        script += "if ! command -v uv &> /dev/null; then\n"
        script += "    curl -LsSf https://astral.sh/uv/install.sh | sh\n"
        script += "    export PATH=\"$HOME/.local/bin:$PATH\"\n"
        script += "fi\n\n"
        
        # Add conda environment activation if specified (for system packages like CUDA)
        env = self.connection.config.get('env')
        if env:
            script += f"# Activate conda environment for system packages (CUDA, etc.)\n"
            script += f"source ~/.bashrc\n"
            script += f"conda activate {env}\n\n"
        
        # Add the actual script content
        script += script_content
        
        return script
    
    def prepare_data(
        self,
        data_source: Union[str, Any],
        source_type: str = "auto",
        local_dir: Optional[str] = None,
        hpc_process: bool = False,
    ) -> str:
        """Prepare data source for job execution.
        
        Args:
            data_source: Data source (path, URL, dataset name, etc.)
            source_type: Type of data source (auto, pdf, iiif, iiif-list, web, directory, hf)
            local_dir: Local directory to save prepared data before upload
            hpc_process: If True, copy raw data to cluster for processing there
            
        Returns:
            Path to data on cluster
        """
        logger.info(f"Preparing data from {source_type} source (hpc_process={hpc_process})...")
        
        # Auto-detect source type if needed
        if source_type == "auto":
            source_type = self._detect_source_type(data_source)
        
        # If hpc_process is enabled, handle data accordingly
        if hpc_process:
            if source_type in ["pdf", "directory"]:
                logger.info("HPC processing enabled - copying raw data to cluster")
                return self._copy_raw_data_to_cluster(data_source, source_type)
            elif source_type in ["iiif", "iiif-list"]:
                logger.info("HPC processing enabled - will download IIIF data on cluster")
                return self._copy_raw_data_to_cluster(data_source, source_type)
        
        # Create dataset from source
        dataset = None
        
        if source_type == "pdf":
            ds = PDFDataSource(data_source)
            dataset = ds.to_dataset()
        elif source_type == "iiif":
            ds = IIIFDataSource(data_source)
            dataset = ds.to_dataset()
        elif source_type == "iiif-list":
            # Load multiple IIIF manifests from a list file
            from datasets import concatenate_datasets
            manifest_urls = self._load_manifest_list(data_source)
            datasets = []
            for url in manifest_urls:
                logger.info(f"Loading IIIF manifest: {url}")
                ds = IIIFDataSource(url)
                datasets.append(ds.to_dataset())
            dataset = concatenate_datasets(datasets)
        elif source_type == "web":
            ds = WebDataSource(data_source)
            dataset = ds.to_dataset()
        elif source_type == "directory":
            ds = DirectoryDataSource(data_source)
            dataset = ds.to_dataset()
        elif source_type == "hf":
            # For HF datasets, we'll just pass the dataset name to the cluster script
            return data_source
        else:
            raise ValueError(f"Unsupported source type: {source_type}")
        
        # Save dataset locally
        if local_dir is None:
            local_dir = tempfile.mkdtemp(prefix="yale_job_")
        
        local_path = os.path.join(local_dir, "dataset")
        logger.info(f"Saving dataset to {local_path}")
        dataset.save_to_disk(local_path)
        
        # Upload to cluster
        remote_path = os.path.join(self.job_dir, f"{self.job_name}_data")
        logger.info(f"Uploading dataset to cluster: {remote_path}")
        
        # Ensure job directory exists
        self.connection.execute_command(f"mkdir -p {self.job_dir}")
        
        # Upload the dataset
        self.connection.upload_directory(local_path, remote_path)
        
        return remote_path
    
    def _load_manifest_list(self, file_path: str) -> List[str]:
        """Load IIIF manifest URLs from a text file.
        
        Args:
            file_path: Path to text file with manifest URLs (one per line)
            
        Returns:
            List of manifest URLs
        """
        with open(file_path, 'r') as f:
            urls = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        return urls
    
    def _copy_raw_data_to_cluster(self, data_source: str, source_type: str) -> str:
        """Copy raw data files to cluster for processing there.
        
        Args:
            data_source: Local path to data or URL/manifest list
            source_type: Type of data source
            
        Returns:
            Path to raw data on cluster (for files) or the data_source itself (for URLs)
        """
        import shutil
        
        # Ensure job directory exists
        self.connection.execute_command(f"mkdir -p {self.job_dir}")
        
        if source_type == "pdf":
            remote_path = os.path.join(self.job_dir, f"{self.job_name}_raw_data")
            # Upload single PDF file
            if os.path.isfile(data_source):
                remote_file = os.path.join(remote_path, os.path.basename(data_source))
                self.connection.execute_command(f"mkdir -p {remote_path}")
                self.connection.upload_file(data_source, remote_file)
                logger.info(f"Uploaded PDF to: {remote_file}")
                return remote_file
            else:
                # Upload directory of PDFs
                self.connection.upload_directory(data_source, remote_path)
                logger.info(f"Uploaded PDF directory to: {remote_path}")
                return remote_path
        elif source_type == "directory":
            remote_path = os.path.join(self.job_dir, f"{self.job_name}_raw_data")
            # Upload entire directory
            self.connection.upload_directory(data_source, remote_path)
            logger.info(f"Uploaded directory to: {remote_path}")
            return remote_path
        elif source_type == "iiif":
            # For IIIF manifests, just return the URL - no upload needed
            logger.info(f"IIIF manifest will be processed on cluster: {data_source}")
            return data_source
        elif source_type == "iiif-list":
            # Upload the manifest list file to cluster
            remote_file = os.path.join(self.job_dir, f"{self.job_name}_manifests.txt")
            self.connection.upload_file(data_source, remote_file)
            logger.info(f"Uploaded manifest list to: {remote_file}")
            return remote_file
        
        raise ValueError(f"Cannot copy raw data for source type: {source_type}")
    
    def _detect_source_type(self, data_source: str) -> str:
        """Auto-detect the type of data source.
        
        Args:
            data_source: Data source path or URL
            
        Returns:
            Detected source type
        """
        if isinstance(data_source, str):
            # Check for manifest list file (.txt with IIIF URLs)
            if data_source.endswith('.txt') and Path(data_source).is_file():
                # Check if file contains IIIF manifest URLs
                with open(data_source, 'r') as f:
                    first_line = f.readline().strip()
                    if first_line.startswith('http') and ('manifest' in first_line or 'iiif' in first_line):
                        return "iiif-list"
            elif data_source.endswith('.pdf') or (Path(data_source).is_dir() and 
                                                 any(Path(data_source).glob('**/*.pdf'))):
                return "pdf"
            elif data_source.startswith('http') and ('manifest' in data_source or 'iiif' in data_source):
                return "iiif"
            elif data_source.startswith('http'):
                return "web"
            elif Path(data_source).is_dir():
                return "directory"
            else:
                # Assume HuggingFace dataset
                return "hf"
        
        return "unknown"
    
    def submit(
        self,
        script_path: str,
        wait: bool = False,
        check_interval: int = 30,
    ) -> str:
        """Submit a job to SLURM.
        
        Args:
            script_path: Path to SLURM batch script on cluster
            wait: Whether to wait for job completion
            check_interval: Seconds between status checks when waiting
            
        Returns:
            Job ID
        """
        import traceback
        logger.info(f"[SUBMIT CALLED] Job: {self.job_name}, Script: {script_path}")
        logger.debug("".join(traceback.format_stack()[-5:]))  # Show call stack
        
        result = self.connection.execute_command(f"sbatch {script_path}")
        
        if result['exit_code'] != 0:
            raise RuntimeError(f"Job submission failed: {result['stderr']}")
        
        # Extract job ID from output (format: "Submitted batch job 12345")
        output = result['stdout'].strip()
        self.job_id = output.split()[-1]
        
        logger.info(f"✓ Job submitted with ID: {self.job_id}")
        
        if wait:
            self.wait_for_completion(check_interval)
        
        return self.job_id
    
    def get_status(self, job_id: Optional[str] = None) -> Dict[str, Any]:
        """Get job status.
        
        Args:
            job_id: Job ID to check (uses self.job_id if not specified)
            
        Returns:
            Dict with job status information
        """
        job_id = job_id or self.job_id
        
        if not job_id:
            raise ValueError("No job ID specified")
        
        # Use sacct for job status
        result = self.connection.execute_command(
            f"sacct -j {job_id} --format=JobID,JobName,State,ExitCode,Elapsed,NodeList -P"
        )
        
        if result['exit_code'] != 0:
            # Try squeue for running jobs
            result = self.connection.execute_command(
                f"squeue -j {job_id} --format='%i|%j|%T|%M|%N' --noheader"
            )
            
            if result['exit_code'] == 0 and result['stdout'].strip():
                lines = result['stdout'].strip().split('\n')
                if lines:
                    fields = lines[0].split('|')
                    return {
                        'job_id': fields[0],
                        'name': fields[1],
                        'state': fields[2],
                        'elapsed': fields[3],
                        'nodes': fields[4],
                    }
        
        # Parse sacct output
        lines = result['stdout'].strip().split('\n')
        if len(lines) > 1:
            # Get the first data line (skip header)
            fields = lines[1].split('|')
            return {
                'job_id': fields[0],
                'name': fields[1],
                'state': fields[2],
                'exit_code': fields[3],
                'elapsed': fields[4],
                'nodes': fields[5],
            }
        
        return {'state': 'UNKNOWN'}
    
    def wait_for_completion(self, check_interval: int = 30):
        """Wait for job to complete.
        
        Args:
            check_interval: Seconds between status checks
        """
        if not self.job_id:
            raise ValueError("No job submitted yet")
        
        logger.info(f"Waiting for job {self.job_id} to complete...")
        
        while True:
            status = self.get_status()
            state = status.get('state', 'UNKNOWN')
            
            logger.info(f"Job status: {state}")
            
            if state in ['COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT']:
                break
            
            time.sleep(check_interval)
        
        if state == 'COMPLETED':
            logger.info("✓ Job completed successfully")
        else:
            logger.warning(f"Job ended with state: {state}")
    
    def get_output(self, output_file: Optional[str] = None) -> str:
        """Get job output log.
        
        Args:
            output_file: Path to output file on cluster
            
        Returns:
            Output log content
        """
        output_file = output_file or f"{self.job_dir}/{self.job_name}.out"
        
        result = self.connection.execute_command(f"cat {output_file}")
        
        if result['exit_code'] != 0:
            raise FileNotFoundError(f"Output file not found: {output_file}")
        
        return result['stdout']
    
    def download_results(self, local_dir: str, result_pattern: str = "*"):
        """Download job results from cluster.
        
        Args:
            local_dir: Local directory to save results
            result_pattern: Pattern for files to download
        """
        logger.info(f"Downloading results to {local_dir}")
        
        # Create local directory
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        
        # List files in result directory
        result = self.connection.execute_command(
            f"find {self.result_dir} -name '{result_pattern}' -type f"
        )
        
        if result['exit_code'] == 0:
            files = result['stdout'].strip().split('\n')
            
            for remote_file in files:
                if remote_file:
                    filename = Path(remote_file).name
                    local_file = os.path.join(local_dir, filename)
                    self.connection.download_file(remote_file, local_file)
        
        logger.info("✓ Results downloaded")
    
    def cancel(self, job_id: Optional[str] = None):
        """Cancel a running job.
        
        Args:
            job_id: Job ID to cancel (uses self.job_id if not specified)
        """
        job_id = job_id or self.job_id
        
        if not job_id:
            raise ValueError("No job ID specified")
        
        logger.info(f"Cancelling job {job_id}")
        
        result = self.connection.execute_command(f"scancel {job_id}")
        
        if result['exit_code'] == 0:
            logger.info("✓ Job cancelled")
        else:
            logger.error(f"Failed to cancel job: {result['stderr']}")

