"""
Utility functions for the medical image classifier
"""

import logging
import json
from pathlib import Path
from typing import Dict, Any

def setup_logging(verbose: bool = False) -> logging.Logger:
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    return logging.getLogger(__name__)

def save_results(data: Dict[Any, Any], output_path: str) -> None:
    """Save classification results to JSON file"""
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Results saved to {output_path}")

def load_results(input_path: str) -> Dict[Any, Any]:
    """Load classification results from JSON file"""
    with open(input_path, 'r') as f:
        return json.load(f)