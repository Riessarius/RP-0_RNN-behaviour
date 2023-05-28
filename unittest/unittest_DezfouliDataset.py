import json
from pathlib import Path

from dataset import DezfouliDataset
from utils import to_rdir

to_rdir()

# ---------------------------- Configuration ----------------------------
config_rdir = Path(r"config/unittest")
# ------------------------------------------------------------------------

print("Dezfouli Dataset - Unit test:")

print("Loading dataset...")
with open(config_rdir / "unittest_DezfouliDataset.json", "r") as f:
    dataset_config = json.load(f)
dataset = DezfouliDataset(**dataset_config)
print("Done!")
print()

print("Basic information:")
print(f"Source Path: {dataset._src_path}; Mode: {dataset._mode}")
print()

print("Original data:")
print(dataset._original_data)
print()

print("Function test: ")

print("Get Length:")
print(len(dataset))
print("Done!")
print()

print("Get Item:")
sample_item = dataset[0:3]
print(f"Input shape:{sample_item[0].shape}; Output shape:{sample_item[1].shape}; Mask shape:{sample_item[2].shape}.")
print(f"Info: {sample_item[3]}.")
print("Done!")
print()

print("Get Subset:")
subset = dataset.subset(list(range(100)))
print(f"Subset length: {len(subset)}")
print("Done!")
print()

print("Get numbers of unique values of info:")
print(dataset.get_info_num_unique())
print("Done!")
print()
