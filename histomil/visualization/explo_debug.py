
# %%
from pathlib import Path

from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from histomil.data.torch_datasets import HDF5WSIDatasetWithTileID
from histomil.models.models import load_model, get_device
from histomil.models.mil_models import AttentionAggregatorPL
from histomil.visualization.heatmap import compute_attention_map

# %%
wsi_dir = Path("/mnt/nas6/data/CPTAC")

# %%
def get_wsi_path(wsi_id, wsi_dir):
    wsi_paths = [f for f in wsi_dir.rglob(wsi_id + ".svs")]
    if len(wsi_paths) > 1:
        raise ValueError(f"Multiple WSI files found for {wsi_id}: {wsi_paths}")
    return wsi_paths[0]

# %%
hdf5_path = "/home/valentin/workspaces/histomil/data/processed/embeddings/superpixels_resnet50__alpha_0.5__ablation.h5"
collate_fn_ragged = HDF5WSIDatasetWithTileID.get_collate_fn_ragged()
val_dataset = HDF5WSIDatasetWithTileID(hdf5_path, split="test")
val_loader = DataLoader(
    val_dataset,
    batch_size=1,
    num_workers=1,
    collate_fn=collate_fn_ragged,
)


# %%
feature_extractor_weights = "/mnt/nas7/data/Personal/Darya/saved_models/superpixels_resnet50__alpha_0.5__ablation_99.pth"
mil_weights = "/home/valentin/workspaces/histomil/models/mil/superpixels_org_alpha0.5_tutobene.ckpt"
device = get_device(gpu_id=0)
mil_aggregator = AttentionAggregatorPL.load_from_checkpoint(mil_weights)

# %%
wsi_ids, embeddings, labels, tile_ids = next(iter(val_loader))

# %%
mil_aggregator.to(device)
mil_aggregator.eval()   

# %%
embedding = embeddings[0].to(device)
pred, proba, attention_scores = mil_aggregator.predict_one_embedding(embedding)

# %%
attention_scores = attention_scores.cpu().numpy()

# %%
attention_scores.min()

# %%
wsi_ids[0]

# %%

tile_ids[0]

# %%
def plot_attention_map(attention_map, thumbnail):
    # Normalize the attention map between 0 and 1
    attention_norm = (attention_map - attention_map.min()) / (
        attention_map.max() - attention_map.min()
    )

    # Plotting
    plt.figure(figsize=(10, 10))

    # Show WSI thumbnail
    plt.imshow(thumbnail, cmap="gray" if thumbnail.ndim == 2 else None)

    # Overlay attention heatmap with transparency
    plt.imshow(attention_norm, cmap="jet", alpha=0.5)  # alpha adjusts transparency

    plt.axis("off")
    plt.title("WSI Thumbnail with Attention Overlay")
    plt.tight_layout()
    plt.show()

# %%
wsi_id = wsi_ids[0]
wsi_path = get_wsi_path(wsi_id, wsi_dir)
attention_map, thumbnail = compute_attention_map(
    attention_scores,
    tile_ids[0],
    tile_size=224,
    tile_mpp=1.0,
    wsi_path=wsi_path,
)

# %%



