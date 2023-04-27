import datetime
import json
from pathlib import Path

import torch
from torch.utils.data import random_split

from agent import RNNAgent
from dataset import DezfouliDataset
from utils import to_rdir

to_rdir()

# ---------------------------- Configuration ----------------------------
torch_seed = 20230307
test_ratio = 0.2
config_rdir = Path(r"config/unittest")
tensorboard_rdir = Path(r"tensorboard/unittest/RNNAgent")
save_rdir = Path(r"save/unittest/RNNAgent")
# ------------------------------------------------------------------------

print("RNN Agent - Unit test:")

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

print("Creating agent...")
with open(config_rdir / "unittest_RNNAgent.json", "r") as f:
    agent_config = json.load(f)
if agent_config["model"]["name"] is None:
    current_time = datetime.datetime.now().strftime("%b%d_%H-%M-%S")
    agent_config["model"]["name"] = current_time
agent = RNNAgent(tensorboard_rdir = tensorboard_rdir, **agent_config["model"])
print("Done!")
print()

print("Basic information:")
print(f"RNN type: {agent._model._rnn_type}; ")
print(f"Input dimension: {agent._model._input_dim}; Hidden dimension: {agent._model._hidden_dim}; Output dimension: {agent._model._output_dim}.\n")
print()

print("Function test:")

test_size = int(len(dataset) * test_ratio)
train_size = len(dataset) - test_size
train_set, test_set = random_split(dataset, [train_size, test_size])

print("Train:")
agent.train(train_set, test_set, **agent_config["training"])
print("Done!")
print()

print("Predict:")
output = agent.predict(test_set, **agent_config["training"])
print(output)
print("Done!")
print()

print("Get Internal State:")
internal_state = agent.get_internal_state()
print(internal_state)
print("Done!")
print()

print("Save:")
save_dir = save_rdir / agent_config["model"]["name"]
agent.save(save_dir)
print("Done!")
print()
