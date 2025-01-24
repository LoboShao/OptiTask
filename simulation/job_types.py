from typing import Callable, Dict, Tuple


def profile_llm(batch_size: int, seq_len: int = 512, hidden: int = 1024, layers: int = 12) -> Tuple[float, float]:
    # Forward pass
    flops_per_token = (
                              4 * seq_len * hidden * hidden +  # Self attention projections
                              seq_len * seq_len * hidden +  # Cross attention
                              8 * seq_len * hidden * hidden  # FFN with 4x expansion
                      ) * layers

    forward_flops = flops_per_token * batch_size

    # Total training FLOPs (forward + backward + optimizer)
    total_flops = forward_flops * 4  # 1x forward + 2x backward + 1x optimizer

    # Memory requirements (activations + gradients + optimizer states)
    memory_gb = (seq_len * hidden * 4 * batch_size * 4) / 1e9  # *4 for fp32 params

    return total_flops * 100, memory_gb * 2  # *100 for additional overhead


def profile_classification(batch_size: int, image_size: int = 224, initial_channels: int = 64) -> Tuple[float, float]:
    # ResNet50 has ~23M parameters and ~4-5B FLOPs per forward pass
    base_flops = 4.5e9  # Base FLOPs for single image forward pass

    forward_flops = base_flops * batch_size
    total_flops = forward_flops * 4  # 1x forward + 2x backward + 1x optimizer

    memory_gb = (image_size * image_size * 3 * 4 * batch_size * 3) / 1e9  # *3 for features

    return total_flops * 10, memory_gb * 2  # *10 for data loading and augmentation overhead


def profile_segmentation(batch_size: int, image_size: int = 512, base_channels: int = 64) -> Tuple[float, float]:
    flops_per_image = (image_size * image_size * 3 * base_channels * 9 +
                       sum((image_size // 2 ** i) ** 2 * base_channels * 2 ** i * base_channels * 9
                           for i in range(3)) * 2)  # encoder + decoder
    forward_flops = flops_per_image * batch_size * 5  # multi-scale
    total_flops = forward_flops * 4  # forward + backward + optimizer
    memory_gb = (image_size * image_size * 4 * batch_size * 3) / 1e9

    return total_flops * 10, memory_gb * 2  # Apply overheads


JOB_PROFILES: Dict[str, Callable[[int], Tuple[float, float]]] = {
    "llm": profile_llm,
    "classification": profile_classification,
    "segmentation": profile_segmentation,
}