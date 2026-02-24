# Utility modules for shared, reusable helper functions and small components used across the project

"""
Utility helpers for the project
"""

import os
import random
import numpy as np

# Create directory
def create_dir(path):
    """Create folder if not exists"""
    os.makedirs(path, exist_ok=True)

# Set global seed
def set_seed(seed=42):
    """Reproducibility helper"""
    random.seed(seed)
    np.random.seed(seed)

# Timestamp string
def timestamp():
    from datetime import datetime
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# Export clean API
__all__ = ["create_dir", "set_seed", "timestamp"]