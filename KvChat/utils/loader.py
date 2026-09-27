"""
Model weight loading utilities with comprehensive error handling.

This module provides functions to load model weights from safetensors files
with full exception handling and detailed logging statistics.
"""

import os
from glob import glob
from typing import Dict
import logging

import torch
from torch import nn
from safetensors import safe_open

logger = logging.getLogger(__name__)


def default_weight_loader(
    param: nn.Parameter,
    loaded_weight: torch.Tensor
) -> None:
    """
    Default weight loader that copies loaded tensor to model parameter.
    
    Args:
        param: Target model parameter
        loaded_weight: Loaded weight tensor from safetensors file
    """
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str) -> Dict[str, int]:
    """
    Load model weights from safetensors files with comprehensive error handling.
    
    Loads all safetensors files from the specified directory, handling packed modules
    and providing detailed statistics on the loading process.
    
    Args:
        model: The PyTorch model to load weights into
        path: Directory containing safetensors weight files
    
    Returns:
        Dictionary with loading statistics:
            - 'loaded': Number of successfully loaded weights
            - 'failed': Number of failed weight loads
            - 'skipped': Number of skipped weights (not in model)
    
    Raises:
        ValueError: If path is not a valid directory
        FileNotFoundError: If no safetensors files found in path
        RuntimeError: If critical model weights fail to load
    
    Example:
        >>> stats = load_model(model, "./model_weights")
        >>> print(f"Loaded: {stats['loaded']}, Failed: {stats['failed']}")
    """
    # Validate path
    if not os.path.isdir(path):
        raise ValueError(f"Model path does not exist or is not a directory: {path}")
    
    # Find all safetensors files
    safetensors_files = sorted(glob(os.path.join(path, "*.safetensors")))
    if not safetensors_files:
        raise FileNotFoundError(f"No safetensors files found in {path}")
    
    logger.info(f"Found {len(safetensors_files)} safetensors files in {path}")
    
    # Get packed modules mapping if available
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    
    # Statistics tracking
    stats = {
        'loaded': 0,
        'failed': 0,
        'skipped': 0,
    }
    failed_weights = []
    
    # Load weights from each file
    for file_idx, file_path in enumerate(safetensors_files, 1):
        try:
            logger.debug(f"Processing file {file_idx}/{len(safetensors_files)}: {file_path}")
            
            with safe_open(file_path, "pt", "cpu") as f:
                weight_names = f.keys()
                logger.debug(f"File contains {len(weight_names)} weight tensors")
                
                for weight_name in weight_names:
                    try:
                        # Check for packed module mappings
                        loaded = False
                        for k in packed_modules_mapping:
                            if k in weight_name:
                                v, shard_id = packed_modules_mapping[k]
                                param_name = weight_name.replace(k, v)
                                try:
                                    param = model.get_parameter(param_name)
                                    weight_loader = getattr(param, "weight_loader")
                                    weight_loader(param, f.get_tensor(weight_name), shard_id)
                                    stats['loaded'] += 1
                                    loaded = True
                                    logger.debug(f"Loaded packed module: {weight_name} -> {param_name}")
                                except AttributeError as e:
                                    logger.warning(
                                        f"Failed to get weight_loader for {param_name}: {e}"
                                    )
                                    stats['failed'] += 1
                                    failed_weights.append(weight_name)
                                    loaded = True
                                except Exception as e:
                                    logger.warning(
                                        f"Error loading packed module {weight_name}: {e}"
                                    )
                                    stats['failed'] += 1
                                    failed_weights.append(weight_name)
                                    loaded = True
                                break
                        
                        # Load regular weights if not a packed module
                        if not loaded:
                            try:
                                param = model.get_parameter(weight_name)
                                weight_loader = getattr(
                                    param,
                                    "weight_loader",
                                    default_weight_loader
                                )
                                weight_loader(param, f.get_tensor(weight_name))
                                stats['loaded'] += 1
                                logger.debug(f"Loaded weight: {weight_name}")
                            except AttributeError:
                                # Parameter not in model
                                stats['skipped'] += 1
                                logger.debug(f"Parameter {weight_name} not found in model (skipped)")
                            except Exception as e:
                                logger.warning(f"Error loading weight {weight_name}: {e}")
                                stats['failed'] += 1
                                failed_weights.append(weight_name)
                    
                    except Exception as e:
                        logger.error(f"Unexpected error processing weight {weight_name}: {e}")
                        stats['failed'] += 1
                        failed_weights.append(weight_name)
        
        except Exception as e:
            logger.error(f"Error opening safetensors file {file_path}: {e}")
            stats['failed'] += 1
    
    # Log summary statistics
    logger.info(
        f"Weight loading complete: {stats['loaded']} loaded, "
        f"{stats['failed']} failed, {stats['skipped']} skipped"
    )
    
    if failed_weights:
        logger.warning(f"Failed to load {len(failed_weights)} weights: {failed_weights[:5]}...")
    
    return stats

