import json
from pathlib import Path

from dataset import DezfouliDataset
from utils import to_rdir

to_rdir()

# ---------------------------- Configuration ----------------------------
config_rdir = Path(r"config/unittest/DezfouliDataset")
# ------------------------------------------------------------------------

print("Dezfouli Dataset - Unit test:")

print("Loading dataset...")
with open(config_rdir / "DezfouliDataset.json", "r") as f:
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
print(f"Get items with slice:")
slice_sample = dataset[:]
print(f"Item keys: {slice_sample.keys()}.")
print(f"Input shape:{slice_sample['input'].shape}; Output shape:{slice_sample['output'].shape}; Mask shape:{slice_sample['mask'].shape}.")
print(f"Get items with indices:")
idx = list(range(12))
idx_sample = dataset[idx]
print(f"Item keys: {idx_sample.keys()}.")
print(f"Input shape:{idx_sample['input'].shape}; Output shape:{idx_sample['output'].shape}; Mask shape:{idx_sample['mask'].shape}.")
print(f"Get items with query:")
query = {
    'subject_no': list(range(12)),
    'block_no': list(range(6)),
}
query_sample = dataset[query]
print(f"Item keys: {query_sample.keys()}.")
print(f"Input shape:{query_sample['input'].shape}; Output shape:{query_sample['output'].shape}; Mask shape:{query_sample['mask'].shape}.")
print("Done!")
print()

print(f"Get values by specifying property:")
prop = 'subject_id'
print(f"Values of property {prop}: {dataset.get_by_prop(prop)}")
print("Done!")
print()

print(f"Subset operations:")
subidx0 = list(range(0, 96, 6))
subset0 = dataset.subset(subidx0)
print(f"Subset 0 has {len(subset0)} items with original indices {subset0._sub_indices}.")
subquery1 = {
    'subject_no': list(range(4)),
    'block_no': list(range(6)),
}
subset1 = subset0.subset(subquery1)
print(f"Subset 1 has {len(subset1)} items with original indices {subset1._sub_indices}.")
subset2 = subset1.subset({})
print(f"Subset 2 has {len(subset2)} items with original indices {subset2._sub_indices}.")
subset3 = subset2.subset([])
print(f"Subset 3 has {len(subset3)} items with original indices {subset3._sub_indices}.")
print("Done!")
print()

print(f"Equality test:")
print(f"Self identity test, which should be True: {dataset == dataset}")
print(f"Test between subset 0 and 1, which should be False: {subset0 == subset1}")
print(f"Test between subset 1 and 2, which should be True: {subset1 == subset2}")
print(f"Test of unexpected argument, which should be False: {dataset == None}")
print("Done!")
print()
