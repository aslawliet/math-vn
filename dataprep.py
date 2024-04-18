from datasets import load_dataset
import random, json

random.seed(42)
dataset_repo_id = "YOUR_HF_DATASET_REPO"
dataset = load_dataset(dataset_repo_id)['train']
dataset = dataset.select(range(1000))
num_rows = dataset.num_rows
print(f"Number of rows: {num_rows}")
validation_ratio = 0.1
dataset = dataset.shuffle()
num_validation_samples = int(num_rows * validation_ratio)
train_size = int((1-validation_ratio) * num_rows)
validation_size = num_rows - train_size

# Split the dataset
train_dataset = dataset.select(range(train_size))
val_dataset = dataset.select(range(train_size, num_rows))

print(f"Train dataset size: {train_size}")
print(f"Validation dataset size: {validation_size}")

print(len(train_dataset))
print(len(val_dataset))

def is_valid_json(json_example):
    try:
        json.dumps(json_example)
        return True
    except (ValueError, TypeError):
        return False

# Save the train dataset in JSONL format
with open("data/train.jsonl", "w") as f:
    for example in train_dataset:
        json_example = {"input": example["input"], "label": example["label"]}
        if is_valid_json(json_example):
            json.dump(json_example, f)
            f.write("\n")
        else:
            print(f"Skipping invalid JSON line: {example}")

# Save the validation dataset in JSONL format
with open("data/validation.jsonl", "w") as f:
    for example in val_dataset:
        json_example = {"input": example["input"], "label": example["label"]}
        if is_valid_json(json_example):
            json.dump(json_example, f)
            f.write("\n")
        else:
            print(f"Skipping invalid JSON line: {example}")
