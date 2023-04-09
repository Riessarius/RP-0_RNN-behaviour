import datetime
import json
from pathlib import Path

import torch

from trainer import CVTrainer
from dataset import DezfouliDataset

# ---------------------------- Configuration ----------------------------
torch_seed = 20230310
config_rdir = Path(r"../../config/unittest")
tensorboard_rdir = Path(r"../../tensorboard/unittest_CVTrainer")
save_rdir = Path(r"../../save/unittest_CVTrainer")
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
if trainer_config["trainer"]["name"] is None:
    current_time = datetime.datetime.now().strftime("%b%d_%H-%M-%S")
    trainer_config["trainer"]["name"] = current_time
trainer = CVTrainer(**trainer_config["trainer"])
print("Done!")
print()

print(f"Function test:")

print(f"Train:")
trainer.train(dataset, trainer_config["agent_model"], trainer_config["agent_training"], tensorboard_rdir = tensorboard_rdir, **trainer_config["trainer_training"])
print("Done!")
print()

print("Save:")
trainer.save(save_rdir = save_rdir)
print("Done!")
print()
