import json
from datasets import load_dataset


def generate_finetune_data():
    print("Loading dataset...")
    # Using the specific config "train" as per your snippet
    dataset = load_dataset("westenfelder/NL2SH-ALFA", "train", split="train")

    total_count = len(dataset)
    print(f"Dataset loaded. Total records: {total_count}")

    # Randomly sample 750 records
    sample_size = 750
    print(f"Randomly sampling {sample_size} records with seed 42...")
    sampled_dataset = dataset.shuffle(seed=42).select(range(sample_size))

    system_prompt = "You are an expert Linux Bash assistant. Translate the user's natural language request into a valid Bash command. Output only the command code without markdown or explanation."
    output_file = "nl2bash_finetune_750.jsonl"

    formatted_data = []

    # Stats counters for diversity check
    stats = {
        "pipes (|)": 0,
        "sudo": 0,
        "awk/sed": 0,
        "find": 0,
        "grep": 0
    }

    for row in sampled_dataset:
        nl_text = row['nl']
        bash_cmd = row['bash']

        if "|" in bash_cmd: stats["pipes (|)"] += 1
        if "sudo" in bash_cmd: stats["sudo"] += 1
        if "awk" in bash_cmd or "sed" in bash_cmd: stats["awk/sed"] += 1
        if "find" in bash_cmd: stats["find"] += 1
        if "grep" in bash_cmd: stats["grep"] += 1

        entry = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": nl_text},
                {"role": "assistant", "content": bash_cmd}
            ]
        }
        formatted_data.append(entry)

    with open(output_file, "w", encoding="utf-8") as f:
        for entry in formatted_data:
            f.write(json.dumps(entry) + "\n")

    print(f"\nSuccessfully generated file: {output_file}")
    print("-" * 30)
    print("Sampled Data Statistics (Diversity Check):")
    for key, val in stats.items():
        print(f"- Contains {key}: {val} records ({val / sample_size:.1%})")
    print("-" * 30)


import json
import random
from datasets import load_dataset


def generate_validation_data():
    print("Loading Test dataset...")
    dataset = load_dataset("westenfelder/NL2SH-ALFA", "test", split="train")

    print(f"Test dataset loaded. Total records: {len(dataset)}")

    # Group by difficulty
    diff_0 = [row for row in dataset if row['difficulty'] == 0]
    diff_1 = [row for row in dataset if row['difficulty'] == 1]
    diff_2 = [row for row in dataset if row['difficulty'] == 2]

    print(f"Pool Stats: Diff_0: {len(diff_0)}, Diff_1: {len(diff_1)}, Diff_2: {len(diff_2)}")

    # Stratified sampling target: 17 + 17 + 16 = 50
    sample_counts = {0: 17, 1: 17, 2: 16}

    # Use fixed seed for reproducibility
    random.seed(42)

    selected_data = []
    selected_data.extend(random.sample(diff_0, sample_counts[0]))
    selected_data.extend(random.sample(diff_1, sample_counts[1]))
    selected_data.extend(random.sample(diff_2, sample_counts[2]))

    # Shuffle the final mix so they aren't ordered by difficulty
    random.shuffle(selected_data)

    # Must use the EXACT same system prompt as the training set
    system_prompt = "You are an expert Linux Bash assistant. Translate the user's natural language request into a valid Bash command. Output only the command code without markdown or explanation."
    output_file = "nl2bash_validation_50.jsonl"

    formatted_lines = []

    for row in selected_data:
        nl_text = row['nl']
        bash_cmd = row['bash']

        entry = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": nl_text},
                {"role": "assistant", "content": bash_cmd}
            ]
        }
        formatted_lines.append(json.dumps(entry))

    with open(output_file, "w", encoding="utf-8") as f:
        for line in formatted_lines:
            f.write(line + "\n")

    print(f"\nSuccessfully generated file: {output_file}")
    print("-" * 30)
    print("Validation Set Composition:")
    final_diffs = [row['difficulty'] for row in selected_data]
    print(f"- Difficulty 0: {final_diffs.count(0)} records")
    print(f"- Difficulty 1: {final_diffs.count(1)} records")
    print(f"- Difficulty 2: {final_diffs.count(2)} records")
    print("-" * 30)



if __name__ == "__main__":
    generate_finetune_data()
    generate_validation_data()