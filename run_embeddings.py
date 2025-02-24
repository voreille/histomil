from pathlib import Path

import subprocess
import torch
import time

# Define list of model weight paths

experiment_names = [
    "superpixel_org", "superpixels_moco_org",
    "superpixels_resnet50__alpha_0.5__ablation"
]

saved_model_dir = Path("/mnt/nas7/data/Personal/Darya/saved_models")

checkpoint_paths = []
for exp_name in experiment_names:
    checkpoint_paths.append(
        list((saved_model_dir / exp_name).glob("*_99.pth"))[0])

print(f"computing the embeddings for {checkpoint_paths}")
for weights, exp_name in zip(checkpoint_paths, experiment_names):
    print(f"Running model with weights: {weights}")

    # Run the Python script as a subprocess
    try:
        process = subprocess.run(
            [
                "python", "histomil/data/compute_embeddings.py",
                "--weights-path",
                str(weights), "--output-filepath",
                f"data/processed/embeddings/{exp_name}.h5", "--num-workers",
                "32", "--gpu-id", "1", "--batch-size", "512", "--max-batches",
                "10"
            ],
            check=True,
        )
    except subprocess.CalledProcessError:
        print(f" Model with weights {weights} failed. Skipping to next model.")
        continue  # Skip to the next model

    print(f"Model with weights {weights} completed.")

    # Clear CUDA memory
    print("Clearing GPU memory...")
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()  # Additional garbage collection for CUDA

    # Small delay to ensure memory cleanup before next iteration
    time.sleep(5)

print("All embeddings computed successfully!")
