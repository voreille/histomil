import hydra
from omegaconf import DictConfig
import torch
import time
from pathlib import Path

@hydra.main(version_base=None, config_path="conf", config_name="config")
def compute_embeddings(cfg: DictConfig) -> None:
    for model_name in cfg.models:
        print(f"Computing embeddings for {model_name}")
        
        output_path = Path(cfg.paths.output_dir) / f"{model_name}_embeddings" / f"{model_name}_cptac.h5"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Import your embedding computation function
            from histomil.data.compute_embeddings import compute_model_embeddings
            
            compute_model_embeddings(
                model_name=model_name,
                output_filepath=str(output_path),
                num_workers=cfg.compute.num_workers,
                gpu_id=cfg.compute.gpu_id,
                batch_size=cfg.compute.batch_size
            )
            
            print(f"Model {model_name} completed successfully")
            
        except Exception as e:
            print(f"Error processing {model_name}: {str(e)}")
            continue
        finally:
            print("Clearing GPU memory...")
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            time.sleep(2)

    print("All embeddings computed successfully!")

if __name__ == "__main__":
    compute_embeddings()