#!/usr/bin/env python
"""
Inspect dataset structure to find the correct field names.
Run on login node (CPU-only).

Usage:
  python scripts/inspect_dataset.py
"""
from datasets import load_dataset

print("Loading OpenR1-Math-220k dataset...")
dataset = load_dataset('open-r1/OpenR1-Math-220k', split='train')

print(f"\nDataset has {len(dataset)} examples")
print(f"Column names: {dataset.column_names}\n")

# Show first 2 examples
print("=" * 80)
print("First example keys and values:")
print("=" * 80)
ex = dataset[0]
for key, val in ex.items():
    val_str = str(val)[:100] if val else "(empty)"
    print(f"  {key}: {val_str}")

print("\n" + "=" * 80)
print("Second example keys and values:")
print("=" * 80)
ex = dataset[1]
for key, val in ex.items():
    val_str = str(val)[:100] if val else "(empty)"
    print(f"  {key}: {val_str}")

# Check coverage of key fields
print("\n" + "=" * 80)
print("Field coverage (first 100 examples):")
print("=" * 80)
sample = dataset.select(range(min(100, len(dataset))))

for col in dataset.column_names:
    non_empty = sum(1 for ex in sample if ex.get(col) and str(ex.get(col)).strip())
    print(f"  {col}: {non_empty}/100 non-empty")
