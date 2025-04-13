import subprocess
import torch
import time

# Define list of model weight paths

# model_names = ["UNI2", "bioptimus"]
model_names = ["UNI2"]
print(f"computing the embeddings for baselines: {model_names}")

for model_name in model_names:
    print(f"Running for {model_name}")

    # Run the Python script as a subprocess
    try:
        process = subprocess.run(
            [
                "python",
                "histomil/data/compute_embeddings.py",
                "--model-name",
                model_name,
                "--output-filepath",
                f"data/interim/embeddings/{model_name}_embeddings/{model_name}_cptac.h5",
                "--num-workers",
                "32",
                "--gpu-id",
                "1",
                "--batch-size",
                "512",
            ],
            check=True,
        )
    except subprocess.CalledProcessError:
        print(f" Model {model_name} failed. Skipping to next model.")
        continue  # Skip to the next model

    print(f"Model {model_name} completed.")

    # Clear CUDA memory
    print("Clearing GPU memory...")
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()  # Additional garbage collection for CUDA

    # Small delay to ensure memory cleanup before next iteration
    time.sleep(5)

print("All embeddings computed successfully!")
