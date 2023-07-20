import datetime
import json
from pathlib import Path

import torch

from dataset import DezfouliDataset
import trainer
from trainer import CVTrainer
from utils import get_num_embeddings, to_rdir

to_rdir()

# ---------------------------- Configuration ----------------------------
torch_seed = 20230310
uid = None
if uid is None:
    uid = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
config_rdir = Path(f"config/unittest/CVTrainer")
tensorboard_rdir = Path(f"tensorboard/unittest/{uid}_CVTrainer")
save_rdir = Path(f"save/unittest/{uid}_CVTrainer")
# ------------------------------------------------------------------------

print("CV Trainer - Unit test:")

print("Setting random seed...")
torch.manual_seed(torch_seed)
print("Done!")
print()

print("Loading dataset...")
with open(config_rdir / "DezfouliDataset.json", "r") as f:
    dataset_config = json.load(f)
dataset = DezfouliDataset(**dataset_config)
print("Done!")
print()

print("Creating trainer...")
with open(config_rdir / "Agent.json", 'r') as f:
    agent_config = json.load(f)
with open(config_rdir / "CVTrainer.json", 'r') as f:
    trainer_config = json.load(f)
if 'embedding_keys' in agent_config['model']['args']:
    agent_config['model']['args']['num_embeddings'] = get_num_embeddings(dataset, agent_config['model']['args']['embedding_keys'])
cv_trainer = CVTrainer(**trainer_config['trainer'])
print("Done!")
print()

print("Function test:")

print("Train:")
cv_trainer.train(dataset, agent_config['model'], agent_config['training'], tensorboard_rdir = tensorboard_rdir, **trainer_config['training'])
print("Done!")
print()

print("Save:")
save_dir = save_rdir / trainer_config['trainer']['name']
cv_trainer.save(save_dir)
print("Done!")
print()

print("Load:")
load_dir = save_dir
new_cv_trainer = trainer.load(load_dir)
print("Done!")
print()
