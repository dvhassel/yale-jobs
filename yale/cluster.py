"""Yale HPC cluster connection and management."""
import os
import time
import yaml
import paramiko
import getpass
from pathlib import Path
from typing import Optional, Dict, Any, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_ssh_config(hostname: str) -> Optional[Dict[str, Any]]:
    """Load SSH config for a given hostname/alias.
    
    Args:
        hostname: Hostname or alias to look up
        
    Returns:
        SSH config dict if found, None otherwise
    """
    try:
        ssh_config = paramiko.SSHConfig()
        config_path = os.path.expanduser('~/.ssh/config')
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                ssh_config.parse(f)
            
            return ssh_config.lookup(hostname)
    except Exception as e:
        logger.debug(f"Could not load SSH config: {e}")
    
    return None


class ClusterConnection:
    """Manage SSH connections to Yale HPC cluster with 2FA support."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize cluster connection.
        
        Args:
            config_path: Path to config.yaml file. If None, looks for config.yaml in current directory
        """
        self.config_path = config_path or "config.yaml"
        self.config = self._load_config()
        self.ssh_client = None
        self.sftp_client = None
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Validate config is a dictionary
        if not isinstance(config, dict):
            raise ValueError(
                f"Invalid config file format. Expected YAML dictionary but got {type(config).__name__}. "
                f"Please check that config.yaml uses 'key: value' format (with colons, not equals)."
            )
        
        # Validate required fields
        required_fields = ['alias']
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required config field: {field}")
        
        return config
    
    def _duo_handler(self, title, instructions, prompt_list):
        """Handle Duo two-factor authentication prompts.
        
        This is called by paramiko during keyboard-interactive auth.
        """
        responses = []
        
        # Display instructions (usually shows Duo options)
        if instructions:
            print(instructions)
        
        for prompt_text, show_input in prompt_list:
            print(prompt_text, end='')
            if show_input:
                response = input()
            else:
                response = getpass.getpass('')
            responses.append(response)
        
        return responses
    
    def connect(self, username: Optional[str] = None, password: Optional[str] = None):
        """Connect to the cluster via SSH.
        
        Args:
            username: Username for SSH connection. If None, uses SSH config or prompts
            password: Password for SSH connection. Not used with publickey+Duo auth
        """
        alias = self.config.get('alias')
        
        # Load SSH config for this alias
        ssh_config = load_ssh_config(alias)
        
        # Get hostname from SSH config or use alias as-is
        hostname = ssh_config.get('hostname', alias) if ssh_config else alias
        
        # Get username: config.yaml > SSH config > provided > prompt
        if username is None:
            username = self.config.get('username')
        if username is None and ssh_config:
            username = ssh_config.get('user')
        if username is None:
            username = input(f"Username for {alias}: ")
        
        logger.info(f"Connecting to {alias} ({hostname})...")
        
        self.ssh_client = paramiko.SSHClient()
        self.ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        
        # Load system host keys
        try:
            self.ssh_client.load_system_host_keys()
        except:
            pass
        
        try:
            logger.info("Connecting with SSH keys and Duo authentication...")
            
            # Get identity file from SSH config if specified
            key_filename = None
            if ssh_config and 'identityfile' in ssh_config:
                identity_files = ssh_config['identityfile']
                if isinstance(identity_files, list):
                    key_filename = [os.path.expanduser(f) for f in identity_files]
                else:
                    key_filename = os.path.expanduser(identity_files)
                logger.info(f"Using identity file(s) from SSH config: {key_filename}")
            
            # Connect using standard SSH client
            self.ssh_client.connect(
                hostname,
                username=username,
                timeout=60,
                look_for_keys=True,
                allow_agent=True,
                auth_timeout=60,
                key_filename=key_filename,
            )
            
            logger.info("SSH key authentication successful")
            
            # Yale requires Duo interaction AFTER successful publickey auth
            # Open an interactive shell to handle the Duo prompt
            logger.info("Handling Duo two-factor authentication...")
            
            import sys
            import select
            
            channel = self.ssh_client.invoke_shell()
            channel.settimeout(0.5)
            
            duo_authenticated = False
            output_buffer = ""
            
            # Interactive loop to handle Duo
            while not duo_authenticated:
                try:
                    # Read from channel
                    if channel.recv_ready():
                        chunk = channel.recv(4096).decode('utf-8', errors='ignore')
                        output_buffer += chunk
                        print(chunk, end='', flush=True)
                        
                        # Check if we got a shell prompt (Duo complete)
                        if any(prompt in output_buffer[-50:] for prompt in ['$ ', '# ', '> ', '] ']):
                            duo_authenticated = True
                            break
                    
                    # Check if user typed something (for Duo response)
                    if sys.platform != 'win32':
                        if select.select([sys.stdin], [], [], 0.0)[0]:
                            user_input = sys.stdin.readline()
                            channel.send(user_input)
                    else:
                        # Windows: use simpler approach
                        import msvcrt
                        if msvcrt.kbhit():
                            user_input = input()
                            channel.send(user_input + '\n')
                
                except Exception as e:
                    # Timeout is expected, continue
                    pass
                
                time.sleep(0.1)
            
            logger.info("Duo authentication successful")
            
            # Send exit to close the shell
            channel.send('exit\n')
            time.sleep(0.5)
            channel.close()
            
            logger.info(f"✓ Successfully connected to {alias}")
            
            # Test connection with a command
            stdin, stdout, stderr = self.ssh_client.exec_command("echo 'Connected'")
            result = stdout.read().decode().strip()
            
            if result == "Connected":
                # Initialize SFTP client
                self.sftp_client = self.ssh_client.open_sftp()
            else:
                raise ConnectionError("Connection test failed")
                
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            if self.ssh_client:
                self.ssh_client.close()
            raise
    
    def execute_command(self, command: str, timeout: int = 30) -> Dict[str, str]:
        """Execute a command on the cluster.
        
        Args:
            command: Command to execute
            timeout: Timeout in seconds
            
        Returns:
            Dict with 'stdout', 'stderr', and 'exit_code' keys
        """
        if not self.ssh_client:
            raise ConnectionError("Not connected to cluster. Call connect() first.")
        
        # Note: We use uv for Python dependency management, so no conda activation needed
        logger.debug(f"Executing: {command}")
        
        stdin, stdout, stderr = self.ssh_client.exec_command(command, timeout=timeout)
        
        stdout_str = stdout.read().decode()
        stderr_str = stderr.read().decode()
        exit_code = stdout.channel.recv_exit_status()
        
        return {
            'stdout': stdout_str,
            'stderr': stderr_str,
            'exit_code': exit_code
        }
    
    def upload_file(self, local_path: str, remote_path: str):
        """Upload a file to the cluster.
        
        Args:
            local_path: Path to local file
            remote_path: Path on cluster
        """
        if not self.sftp_client:
            raise ConnectionError("SFTP not initialized. Call connect() first.")
        
        logger.info(f"Uploading {local_path} to {remote_path}")
        
        # Ensure remote directory exists
        remote_dir = str(Path(remote_path).parent)
        self._ensure_remote_directory(remote_dir)
        
        self.sftp_client.put(local_path, remote_path)
        logger.info("✓ Upload complete")
    
    def upload_directory(self, local_dir: str, remote_dir: str):
        """Upload a directory to the cluster recursively.
        
        Args:
            local_dir: Path to local directory
            remote_dir: Path on cluster
        """
        if not self.sftp_client:
            raise ConnectionError("SFTP not initialized. Call connect() first.")
        
        local_path = Path(local_dir)
        logger.info(f"Uploading directory: {local_path} -> {remote_dir}")
        
        # First, ensure the base remote directory exists
        self._ensure_remote_directory(remote_dir)
        
        # Collect all unique subdirectories that need to be created
        subdirs = set()
        files_to_upload = []
        
        for item in local_path.rglob('*'):
            if item.is_file():
                rel_path = item.relative_to(local_path)
                remote_file = os.path.join(remote_dir, str(rel_path))
                remote_file_dir = str(Path(remote_file).parent)
                
                if remote_file_dir != remote_dir:
                    subdirs.add(remote_file_dir)
                
                files_to_upload.append((str(item), remote_file))
        
        # Create all subdirectories first
        logger.info(f"Creating {len(subdirs)} subdirectories...")
        for subdir in sorted(subdirs):
            self._ensure_remote_directory(subdir)
        
        # Upload all files
        logger.info(f"Uploading {len(files_to_upload)} files...")
        for local_file, remote_file in files_to_upload:
            try:
                self.sftp_client.put(local_file, remote_file)
            except Exception as e:
                logger.error(f"Failed to upload {local_file} to {remote_file}: {e}")
                raise
    
    def download_file(self, remote_path: str, local_path: str):
        """Download a file from the cluster.
        
        Args:
            remote_path: Path on cluster
            local_path: Path to save locally
        """
        if not self.sftp_client:
            raise ConnectionError("SFTP not initialized. Call connect() first.")
        
        logger.info(f"Downloading {remote_path} to {local_path}")
        
        # Ensure local directory exists
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        
        self.sftp_client.get(remote_path, local_path)
        logger.info("✓ Download complete")
    
    def _ensure_remote_directory(self, remote_dir: str):
        """Ensure a remote directory exists, creating it if necessary.
        
        Args:
            remote_dir: Remote directory path
        """
        try:
            self.sftp_client.stat(remote_dir)
            # Directory exists, we're good
            return
        except FileNotFoundError:
            pass  # Directory doesn't exist, need to create it
        
        # Create directory via SSH
        result = self.execute_command(f"mkdir -p '{remote_dir}' && echo 'SUCCESS'")
        
        # Check if command succeeded
        if result['exit_code'] != 0:
            raise RuntimeError(f"Failed to create directory: {remote_dir}. Error: {result['stderr']}")
        
        if 'SUCCESS' not in result['stdout']:
            raise RuntimeError(f"Failed to create directory: {remote_dir}. No SUCCESS in output")
        
        # Verify it was created by trying to stat it again
        max_retries = 3
        for i in range(max_retries):
            try:
                self.sftp_client.stat(remote_dir)
                return  # Success!
            except FileNotFoundError:
                if i < max_retries - 1:
                    import time
                    time.sleep(0.5)  # Wait a bit and retry
                else:
                    raise RuntimeError(f"Directory created but not accessible: {remote_dir}")
    
    def list_directory(self, remote_dir: str) -> List[str]:
        """List contents of a remote directory.
        
        Args:
            remote_dir: Remote directory path
            
        Returns:
            List of filenames
        """
        if not self.sftp_client:
            raise ConnectionError("SFTP not initialized. Call connect() first.")
        
        return self.sftp_client.listdir(remote_dir)
    
    def close(self):
        """Close the SSH connection."""
        if self.sftp_client:
            self.sftp_client.close()
            self.sftp_client = None
        
        if self.ssh_client:
            self.ssh_client.close()
            self.ssh_client = None
        
        logger.info("Connection closed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

