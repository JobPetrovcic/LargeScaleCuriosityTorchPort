import os
import sys
import logging
import wandb
from typing import Dict, Any, Optional

_logger = logging.getLogger("rl_logger")
_logger.setLevel(logging.INFO)
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
_logger.addHandler(_handler)

_log_dir: Optional[str] = None
_kvs: Dict[str, Any] = {}

def configure(dir: str, format_strs: Optional[list] = None):
    global _log_dir
    _log_dir = dir
    os.makedirs(_log_dir, exist_ok=True)
    
    # Add file handler
    file_handler = logging.FileHandler(os.path.join(_log_dir, "log.txt"))
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    _logger.addHandler(file_handler)

def get_dir() -> Optional[str]:
    return _log_dir

def info(msg: str, *args):
    _logger.info(msg, *args)

def logkvs(kvs: Dict[str, Any]):
    global _kvs
    _kvs.update(kvs)

def dumpkvs():
    global _kvs
    # Log to wandb
    if wandb.run is not None:
        wandb.log(_kvs)
    
    # Log to console/file
    kv_str = " | ".join([f"{k}: {v}" for k, v in _kvs.items()])
    info(kv_str)
    
    _kvs = {}

def log(msg: str, *args):
    info(msg, *args)
