import datetime
import json
from pathlib import Path

import torch

from dataset import DezfouliDataset
from trainer import CVTrainer
from utils import get_num_embeddings, to_rdir

to_rdir()

# ---------------------------- Configuration ----------------------------
torch_seed = 20230310
uid = None
if uid is None:
    uid = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
config_rdir = Path(f"config/unittest")
tensorboard_rdir = Path(f"tensorboard/unittest/{uid}_CVTrainer")
save_rdir = Path(f"save/unittest/{uid}_CVTrainer")
# ------------------------------------------------------------------------

print("CV Trainer - Unit test:")

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
with open(config_rdir / "unittest_CVTrainer.json", 'r') as f:
    trainer_config = json.load(f)
if 'embedding_keys' in trainer_config['agent_model']['args']:
    trainer_config['agent_model']['args']['num_embeddings'] = get_num_embeddings(dataset, trainer_config['agent_model']['args']['embedding_keys'])
trainer = CVTrainer(**trainer_config['trainer'])
print("Done!")
print()

print("Function test:")

print("Train:")
trainer.train(dataset, trainer_config['agent_model'], trainer_config['agent_training'], tensorboard_rdir = tensorboard_rdir, **trainer_config['trainer_training'])
print("Done!")
print()

print("Save:")
save_dir = save_rdir / trainer_config['trainer']['name']
trainer.save(save_dir)
print("Done!")
print()
