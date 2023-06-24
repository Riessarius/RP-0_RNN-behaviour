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
print(f"Obtain specific property:")
idx = 'subject_id'
print(f"Values of property {idx}: {dataset[idx]}.")
print(f"Obtain data with indices:")
idx = list(range(10))
idx_sample = dataset[idx]
print(f"Item keys: {idx_sample.keys()}")
print(f"Input shape:{idx_sample['input'].shape}; Output shape:{idx_sample['output'].shape}; Mask shape:{idx_sample['mask'].shape}.")
print(f"Obtain data with criteria:")
query = {
    'subject_no': list(range(10)),
    'block_no': list(range(10)),
}
query_sample = dataset[query]
print(f"Item keys: {query_sample.keys()}")
print(f"Input shape:{query_sample['input'].shape}; Output shape:{query_sample['output'].shape}; Mask shape:{query_sample['mask'].shape}.")
print("Done!")
print()

