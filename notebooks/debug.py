# %%
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from histomil.models.mil_models import AttentionAggregatorPL
from histomil.training.utils import get_device
from histomil.data.torch_datasets import HDF5WSIDataset, collate_fn_ragged

# %%
model = AttentionAggregatorPL.load_from_checkpoint(
    "/home/valentin/workspaces/histomil/models/test_checkpoint_bceloss_first.ckpt",
    input_dim=2048,
)

# %%
gpu_id = 0
device = get_device(gpu_id)

# %%
model.to(device)
model.eval()

# %%
test_dataset = HDF5WSIDataset(
    "/home/valentin/workspaces/histomil/data/processed/embeddings/superpixel_org_copy.h5",
    split="test")
test_loader = DataLoader(
    test_dataset,
    batch_size=128,
    num_workers=0,
    collate_fn=collate_fn_ragged,
)


# %%
trainer = pl.Trainer(accelerator="gpu", devices=[gpu_id])

# %%
results = trainer.predict(model, dataloaders=test_loader)

# %%



