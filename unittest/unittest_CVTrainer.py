import datetime
import json
from pathlib import Path

import torch

from dataset import DezfouliDataset
from trainer import CVTrainer
from utils import to_rdir

to_rdir()

# ---------------------------- Configuration ----------------------------
torch_seed = 20230310
config_rdir = Path(r"config/unittest")
tensorboard_rdir = Path(r"tensorboard/unittest/CVTrainer")
save_rdir = Path(r"save/unittest")
# ------------------------------------------------------------------------

print(f"CV Trainer - Unit test:")

print("Setting random seed...")
torch.manual_seed(torch_seed)
print("Done!")
print()

print("Loading dataset...")
with open(config_rdir / "unittest_DezfouliDataset.json", "r") as f:
    dataset_config = json.load(f)
dataset = DezfouliDataset(**dataset_config)
print("Done!")
print()

print("Creating trainer...")
with open(config_rdir / "unittest_CVTrainer.json", "r") as f:
    trainer_config = json.load(f)
if "embedding_keys" in trainer_config["agent_model"]["args"]:
    num_unique = dataset.get_info_num_unique()
    trainer_config["agent_model"]["args"]["num_embeddings"] = []
    for k in trainer_config["agent_model"]["args"]["embedding_keys"]:
        trainer_config["agent_model"]["args"]["num_embeddings"].append(num_unique[k])
trainer = CVTrainer(**trainer_config["trainer"])
print("Done!")
print()

print(f"Function test:")

print(f"Train:")
trainer.train(dataset, trainer_config["agent_model"], trainer_config["agent_training"], tensorboard_rdir = tensorboard_rdir, **trainer_config["trainer_training"])
print("Done!")
print()

print("Save:")
save_dir = save_rdir / trainer_config["trainer"]["name"]
trainer.save(save_dir)
print("Done!")
print()
