#!/usr/bin/env python3
"""
Environment variable loader utility.

This module provides functions to load environment variables from .env files
and validate required tokens for the Ensemble CAM project.
"""

import os
import sys
from pathlib import Path
from typing import Optional

def load_env_file(env_file_path: Optional[Path] = None) -> bool:
    """
    Load environment variables from .env file.
    
    Args:
        env_file_path: Path to .env file. If None, looks for .env in project root.
        
    Returns:
        bool: True if .env file was found and loaded, False otherwise.
    """
    if env_file_path is None:
        # Look for .env file in project root (3 levels up from this file)
        env_file_path = Path(__file__).parent.parent.parent / '.env'
    
    if not env_file_path.exists():
        return False
    
    try:
        with open(env_file_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue
                
                # Parse key=value pairs
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # Remove quotes if present
                    if value.startswith('"') and value.endswith('"'):
                        value = value[1:-1]
                    elif value.startswith("'") and value.endswith("'"):
                        value = value[1:-1]
                    
                    os.environ[key] = value
                else:
                    print(f"Warning: Invalid line {line_num} in {env_file_path}: {line}")
        
        return True
    except Exception as e:
        print(f"Error loading .env file {env_file_path}: {e}")
        return False

def get_huggingface_token() -> Optional[str]:
    """
    Get Hugging Face token from environment variables.
    
    Returns:
        str: Hugging Face token if found, None otherwise.
    """
    return os.getenv("HUGGINGFACE_HUB_TOKEN")

def get_github_token() -> Optional[str]:
    """
    Get GitHub token from environment variables.
    
    Returns:
        str: GitHub token if found, None otherwise.
    """
    return os.getenv("GITHUB_TOKEN")

def validate_tokens() -> bool:
    """
    Validate that required tokens are available.
    
    Returns:
        bool: True if all required tokens are available, False otherwise.
    """
    hf_token = get_huggingface_token()
    github_token = get_github_token()
    
    missing_tokens = []
    
    if not hf_token or hf_token == "your_huggingface_token_here":
        missing_tokens.append("HUGGINGFACE_HUB_TOKEN")
    
    if not github_token or github_token == "your_github_token_here":
        missing_tokens.append("GITHUB_TOKEN (optional)")
    
    if missing_tokens:
        print("Missing or placeholder tokens found:")
        for token in missing_tokens:
            print(f"  - {token}")
        print("\nPlease:")
        print("1. Copy .env.example to .env")
        print("2. Edit .env and replace placeholder values with your actual tokens")
        print("3. Or set environment variables manually")
        return False
    
    return True

def setup_environment() -> bool:
    """
    Setup environment by loading .env file and validating tokens.
    
    Returns:
        bool: True if environment is properly configured, False otherwise.
    """
    # Load .env file
    env_loaded = load_env_file()
    
    if env_loaded:
        print("Environment variables loaded from .env file")
    else:
        print("No .env file found, using system environment variables")
    
    # Validate tokens
    return validate_tokens()

if __name__ == "__main__":
    """Test the environment loader."""
    print("Testing environment loader...")
    
    if setup_environment():
        print("Environment setup successful!")
        print(f"Hugging Face token: {'✓' if get_huggingface_token() else '✗'}")
        print(f"GitHub token: {'✓' if get_github_token() else '✗'}")
    else:
        print("Environment setup failed!")
        sys.exit(1)
