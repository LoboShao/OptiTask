import numpy as np
from typing import Dict


# def generate_llm_job_params() -> Dict:
#     """Generate realistic parameters for LLM training jobs"""
#     # Dataset sizes (number of examples)
#     dataset_sizes = [
#         10_000,  # Small fine-tuning
#         50_000,  # Medium fine-tuning
#         100_000,  # Large fine-tuning
#         500_000,  # Small pretraining
#         1_000_000,  # Medium pretraining
#         5_000_000,  # Large pretraining
#     ]
#
#     # Batch sizes commonly used (adjusted based on GPU memory)
#     batch_sizes = [8, 16, 32, 64, 128, 256]
#
#     # Sequence lengths
#     seq_lengths = [512, 1024, 2048, 4096]
#
#     # Model sizes (hidden dimension)
#     hidden_dims = [768, 1024, 2048, 4096]
#
#     # Number of layers
#     layer_counts = [6, 12, 24, 36]
#
#     # Sample parameters with different probabilities
#     dataset_size = np.random.choice(dataset_sizes, p=[0.2, 0.2, 0.25, 0.2, 0.1, 0.05])
#     batch_size = np.random.choice(batch_sizes, p=[0.1, 0.2, 0.3, 0.2, 0.15, 0.05])
#     seq_len = np.random.choice(seq_lengths, p=[0.4, 0.3, 0.2, 0.1])
#     hidden = np.random.choice(hidden_dims, p=[0.3, 0.4, 0.2, 0.1])
#     layers = np.random.choice(layer_counts, p=[0.2, 0.4, 0.3, 0.1])
#
#     # Epoch count tends to be smaller for larger datasets
#     if dataset_size < 100_000:
#         epochs = np.random.randint(3, 10)
#     elif dataset_size < 1_000_000:
#         epochs = np.random.randint(2, 5)
#     else:
#         epochs = np.random.randint(1, 3)
#
#     return {
#         "batch_size": batch_size,
#         "dataset_size": dataset_size,
#         "epochs": epochs,
#         "seq_len": seq_len,
#         "hidden": hidden,
#         "layers": layers
#     }

def generate_llm_job_params() -> Dict:
    """Generate realistic parameters for LLM training jobs with interdependencies,
    biased towards larger models."""
    # Define possible values
    dataset_sizes = [50_000, 100_000, 500_000, 1_000_000, 5_000_000]
    batch_sizes = [8, 16, 32, 64, 128, 256]
    seq_lengths = [512, 1024, 2048, 4096]
    hidden_dims = [768, 1024, 2048, 4096]
    layer_counts = [6, 12, 24, 36]

    # Sample dataset size with custom probabilities
    dataset_size = np.random.choice(dataset_sizes, p=[0.2, 0.3, 0.25, 0.15, 0.1])

    # Sample model size parameters with a bias toward larger models:
    # Favor higher hidden dimensions: 768 (10%), 1024 (20%), 2048 (40%), 4096 (30%)
    hidden = np.random.choice(hidden_dims, p=[0.1, 0.2, 0.4, 0.3])
    # Favor deeper networks: 6 (10%), 12 (20%), 24 (40%), 36 (30%)
    layers = np.random.choice(layer_counts, p=[0.1, 0.2, 0.4, 0.3])

    # Adjust batch size based on hidden dimensions: larger models get smaller batches
    if hidden >= 2048:
        possible_batches = [b for b in batch_sizes if b <= 64]
    else:
        possible_batches = batch_sizes
    batch_size = np.random.choice(possible_batches)

    # Sample sequence length
    seq_len = np.random.choice(seq_lengths, p=[0.4, 0.3, 0.2, 0.1])

    # Epoch count tuned to dataset size
    if dataset_size < 100_000:
        epochs = np.random.randint(3, 10)
    elif dataset_size < 1_000_000:
        epochs = np.random.randint(2, 5)
    else:
        epochs = np.random.randint(1, 3)

    return {
        "batch_size": batch_size,
        "dataset_size": dataset_size,
        "epochs": epochs,
        "seq_len": seq_len,
        "hidden": hidden,
        "layers": layers
    }




def generate_classification_job_params() -> Dict:
    """Generate realistic parameters for image classification jobs"""
    # Dataset sizes for classification
    dataset_sizes = [1_000, 5_000, 10_000, 50_000, 100_000, 1_000_000]

    # Common batch sizes for CV tasks
    batch_sizes = [16, 32, 64, 128, 256, 512]

    # Common image sizes
    image_sizes = [128, 224, 256, 320, 384, 512]

    # Initial channel counts
    initial_channels = [32, 64, 96, 128]

    # Sample with probabilities
    dataset_size = np.random.choice(dataset_sizes, p=[0.1, 0.15, 0.25, 0.2, 0.2, 0.1])
    batch_size = np.random.choice(batch_sizes, p=[0.05, 0.15, 0.25, 0.25, 0.2, 0.1])
    image_size = np.random.choice(image_sizes, p=[0.05, 0.3, 0.3, 0.15, 0.15, 0.05])
    initial_channel = np.random.choice(initial_channels, p=[0.1, 0.4, 0.3, 0.2])

    # Epochs based on dataset size
    if dataset_size < 10_000:
        epochs = np.random.randint(50, 200)
    elif dataset_size < 100_000:
        epochs = np.random.randint(20, 100)
    else:
        epochs = np.random.randint(10, 50)

    return {
        "batch_size": batch_size,
        "dataset_size": dataset_size,
        "epochs": epochs,
        "image_size": image_size,
        "initial_channels": initial_channel
    }


def generate_segmentation_job_params() -> Dict:
    """Generate realistic parameters for image segmentation jobs"""
    # Dataset sizes for segmentation (typically smaller than classification)
    dataset_sizes = [500, 1_000, 5_000, 10_000, 50_000, 100_000]

    # Common batch sizes (smaller due to memory requirements)
    batch_sizes = [4, 8, 16, 32, 64, 128]

    # Common image sizes (usually larger for segmentation)
    image_sizes = [256, 384, 512, 768, 1024]

    # Base channel counts
    base_channels = [32, 48, 64, 96, 128]

    # Sample with probabilities
    dataset_size = np.random.choice(dataset_sizes, p=[0.1, 0.15, 0.25, 0.25, 0.15, 0.1])
    batch_size = np.random.choice(batch_sizes, p=[0.1, 0.2, 0.3, 0.2, 0.15, 0.05])
    image_size = np.random.choice(image_sizes, p=[0.15, 0.2, 0.3, 0.25, 0.1])
    base_channel = np.random.choice(base_channels, p=[0.1, 0.2, 0.4, 0.2, 0.1])

    # Epochs based on dataset size
    if dataset_size < 5_000:
        epochs = np.random.randint(50, 200)
    elif dataset_size < 50_000:
        epochs = np.random.randint(20, 100)
    else:
        epochs = np.random.randint(10, 50)

    return {
        "batch_size": batch_size,
        "dataset_size": dataset_size,
        "epochs": epochs,
        "image_size": image_size,
        "base_channels": base_channel
    }