from typing import Callable, Dict, Tuple

from typing import Tuple
import numpy as np


def profile_llm(batch_size: int, dataset_size: int, epochs: int = 3,
                seq_len: int = 512, hidden: int = 1024, layers: int = 12) -> Tuple[float, float]:
    """
    Calculate compute and memory requirements for training an LLM.

    Args:
        batch_size: Batch size for training
        dataset_size: Number of examples in the dataset
        epochs: Number of training epochs
        seq_len: Sequence length for each example
        hidden: Hidden dimension size
        layers: Number of transformer layers

    Returns:
        Tuple[float, float]: (total_petaflops, total_memory_gb)
    """
    # Calculate iterations per epoch
    iterations_per_epoch = dataset_size // batch_size

    # Single token computations for a transformer layer
    flops_per_token = (
        # Self-attention block
            4 * hidden * hidden +  # Query, Key, Value projections + output projection
            seq_len * hidden +  # Attention softmax and dropout
            # Cross-attention computation
            2 * seq_len * hidden +  # Q*K^T matrix multiply
            seq_len * hidden +  # Attention * V
            # FFN block (with 4x expansion)
            2 * 4 * hidden * hidden  # Two linear layers with 4x expansion
    )

    # Multiply by sequence length and layers
    total_flops_per_token = flops_per_token * seq_len * layers

    # Forward pass for the batch
    forward_flops_per_batch = total_flops_per_token * batch_size

    # Training multiplier (forward + backward + optimizer)
    training_multiplier = 4.0

    # Total compute across all iterations and epochs
    total_compute_flops = forward_flops_per_batch * training_multiplier * iterations_per_epoch * epochs

    # Data loading and preprocessing FLOPs (typically much smaller)
    data_loading_flops = dataset_size * seq_len * 100  # Tokenization and preprocessing

    # Total FLOPs
    total_flops = total_compute_flops + data_loading_flops


    # MEMORY CALCULATIONS

    # Model parameters (4 bytes per parameter)
    params_per_layer = 4 * hidden * hidden  # Attention matrices
    params_per_layer += 4 * hidden * hidden  # FFN with 4x expansion factor
    params_per_layer += 4 * hidden  # Layer norms

    total_params = params_per_layer * layers
    total_params += 2 * hidden * seq_len  # Embeddings + positional encodings

    model_memory_gb = (total_params * 4) / 1e9  # 4 bytes per parameter

    # Activations memory (forward pass)
    activations_per_layer = batch_size * seq_len * hidden
    activations_memory_gb = (activations_per_layer * layers * 4) / 1e9

    # Gradient memory (backward pass)
    gradient_memory_gb = activations_memory_gb

    # Optimizer states (Adam uses 2 additional states per parameter)
    optimizer_memory_gb = model_memory_gb * 2

    # Dataset memory (with improved calculation)
    bytes_per_token = 4  # float32
    additional_memory_per_example = seq_len * 2  # For masks, indices, etc.
    dataset_memory_gb = (dataset_size * (seq_len * bytes_per_token + additional_memory_per_example)) / 1e9

    # Total memory for training
    total_memory_gb = (
            model_memory_gb +  # Model parameters
            activations_memory_gb +  # Forward activations
            gradient_memory_gb +  # Gradients
            optimizer_memory_gb +  # Optimizer states
            dataset_memory_gb  # Dataset in memory
    )

    # Apply a memory overhead factor
    total_memory_gb *= 1.5
    return total_flops, total_memory_gb


def profile_classification(batch_size: int, dataset_size: int, epochs: int = 100,
                           image_size: int = 224, initial_channels: int = 64) -> Tuple[float, float]:
    """
    Calculate compute and memory requirements for training a classification model.

    Returns:
        Tuple[float, float]: (total_tflops, total_memory_gb)
    """
    iterations_per_epoch = dataset_size // batch_size

    # ResNet50 computations
    base_flops = 4.5e9  # Base FLOPs for single image forward pass
    forward_flops_per_batch = base_flops * batch_size

    # Total training FLOPs
    training_multiplier = 4.0  # Forward + backward + optimizer steps
    total_compute_flops = forward_flops_per_batch * training_multiplier * iterations_per_epoch * epochs

    # Data loading and augmentation FLOPs
    data_loading_flops = dataset_size * image_size * image_size * 3 * 10  # IO and augmentation

    total_flops = total_compute_flops + data_loading_flops

    # Memory requirements
    model_memory = 23e6 * 4 / 1e9  # ~23M parameters
    batch_memory = (image_size * image_size * 3 * 4 * batch_size * 3) / 1e9
    dataset_memory = (dataset_size * image_size * image_size * 3 * 4) / 1e9
    optimizer_memory = model_memory * 2

    total_memory_gb = model_memory + batch_memory + dataset_memory + optimizer_memory
    return total_flops, total_memory_gb


def profile_segmentation(batch_size: int, dataset_size: int, epochs: int = 100,
                         image_size: int = 512, base_channels: int = 64) -> Tuple[float, float]:
    """
    Calculate compute and memory requirements for training a segmentation model.

    Returns:
        Tuple[float, float]: (total_tflops, total_memory_gb)
    """
    iterations_per_epoch = dataset_size // batch_size

    # U-Net style computations - calculate each level separately to avoid overflow
    level_flops = []

    # Input level
    level_flops.append(float(image_size * image_size * 3 * base_channels * 9))

    # Encoder-decoder levels
    for i in range(3):
        curr_size = image_size // (2 ** i)
        curr_channels = base_channels * (2 ** i)
        level_flop = float(curr_size * curr_size * curr_channels * curr_channels * 9)
        level_flops.append(level_flop)

    # Sum all levels and double for encoder-decoder
    flops_per_image = sum(level_flops) * 2

    # Multi-scale factor
    forward_flops_per_batch = flops_per_image * batch_size * 5

    # Total training FLOPs
    training_multiplier = 4.0  # Forward + backward + optimizer steps
    total_compute_flops = forward_flops_per_batch * training_multiplier * iterations_per_epoch * epochs

    # Data loading and preprocessing FLOPs
    data_loading_flops = dataset_size * image_size * image_size * 3 * 20  # Mask processing overhead

    total_flops = total_compute_flops + data_loading_flops

    # Memory requirements
    channel_sum = sum(base_channels * (2 ** i) for i in range(5))
    model_memory = (channel_sum * 4 * 4) / 1e9
    batch_memory = (image_size * image_size * 4 * batch_size * 3) / 1e9
    dataset_memory = (dataset_size * image_size * image_size * 4 * 2) / 1e9  # Images + masks
    optimizer_memory = model_memory * 2

    total_memory_gb = model_memory + batch_memory + dataset_memory + optimizer_memory

    return total_flops, total_memory_gb
