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
print(f"Mode: {dataset._mode}; Source Path: {dataset._src_path}.")
print()

print("Original data:")
print(dataset._original_data)
print()

print("Function test: ")

print("Mode operation:")
dataset.add_mode("test", {"input": "PLACEHOLDER OF REPLACEMENT"})
dataset.set_mode("test")
print(f"This property should be replaced: {dataset._data['input']}.")
print(f"This property should remain the same as default: {dataset._data['output']}.")
dataset.remove_mode("test")
print("Done!")

print("Get Length:")
print(len(dataset))
print("Done!")
print()

print("Get Item:")
sample_item = dataset[list(range(10))]
print(f"Item keys: {sample_item.keys()}")
print(f"Input shape:{sample_item['input'].shape}; Output shape:{sample_item['output'].shape}; Mask shape:{sample_item['mask'].shape}.")
print("Done!")
print()

print("Get Subset:")
subset = dataset.subset(list(range(100)))
print(f"Subset length: {len(subset)}")
print("Done!")
print()

print("Get numbers of unique values of each property:")
print(dataset.get_num_unique())
print("Done!")
print()
