import random

import json
import subprocess
import shutil
from datasets import load_dataset


if not shutil.which("shellcheck"):
    raise EnvironmentError(
        "Error: 'shellcheck' not found. Please install it via 'apt install shellcheck' or 'brew install shellcheck'.")


def is_code_safe_by_shellcheck(bash_cmd):
    full_script = f"#!/bin/bash\n{bash_cmd}"

    try:
        proc = subprocess.run(
            ["shellcheck", "-s", "bash", "--format=json", "-"],
            input=full_script,
            capture_output=True,
            text=True
        )
    except Exception as e:
        print(f"Shellcheck execution failed: {e}")
        return False

    if proc.returncode != 0 and not proc.stdout:
        return False

    try:
        issues = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return False

    if not issues:
        return True

    for issue in issues:
        level = issue.get('level')  # 'error', 'warning', 'info', 'style'
        code = issue.get('code')

        if level in ['error', 'warning']:
            return False

    return True


def generate_finetune_data(ofile = None):
    print("Loading dataset...")
    dataset = load_dataset("westenfelder/NL2SH-ALFA", "train", split="train")

    target_count = 1000

    shuffled_dataset = dataset.shuffle(seed = 114514)

    system_prompt = "You are an expert Linux Bash assistant. Translate the user's natural language request into a valid Bash command. Output only the command code without markdown or explanation."
    output_file = f"nl2bash_finetune_{target_count}.jsonl" if not ofile else ofile

    formatted_data = []
    stats = {"scanned": 0, "kept": 0, "rejected": 0}

    print(f"Starting ShellCheck scan... Target: {target_count} high-quality records.")

    for row in shuffled_dataset:
        if stats["kept"] >= target_count:
            break

        stats["scanned"] += 1
        nl_text = row['nl']
        bash_cmd = row['bash']

        if is_code_safe_by_shellcheck(bash_cmd):
            entry = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": nl_text},
                    {"role": "assistant", "content": bash_cmd}
                ]
            }
            formatted_data.append(entry)
            stats["kept"] += 1
        else:
            stats["rejected"] += 1

        if stats["scanned"] % 100 == 0:
            pass_rate = stats["kept"] / stats["scanned"]
            print(f"\rScanned: {stats['scanned']} | Kept: {stats['kept']} | Pass Rate: {pass_rate:.1%}", end="")

    print(f"\n\nDone!")
    if stats["kept"] < target_count:
        print(f"Warning: Only found {stats['kept']} valid records after scanning entire dataset.")

    with open(output_file, "w", encoding="utf-8") as f:
        for entry in formatted_data:
            f.write(json.dumps(entry) + "\n")

    print(f"Successfully generated file: {output_file}")
    print("-" * 30)
    print(f"Final Filter Stats:")
    print(f"- Total Scanned: {stats['scanned']}")
    print(f"- Kept (Safe):   {stats['kept']}")
    print(f"- Rejected:      {stats['rejected']}")
    print(f"- Final Pass Rate: {stats['kept'] / stats['scanned']:.1%}")
    print("-" * 30)


def generate_validation_data(ofile = None):
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
    output_file = "nl2bash_validation_50.jsonl" if not ofile else ofile

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


def generate_eval_data(ofile = None, n = 50, seed = 114514):
    print("Loading Test dataset...")
    dataset = load_dataset("westenfelder/NL2SH-ALFA", "test", split="train")

    print(f"Test dataset loaded. Total records: {len(dataset)}")

    # Group by difficulty
    diff_0 = [row for row in dataset if row['difficulty'] == 0]
    diff_1 = [row for row in dataset if row['difficulty'] == 1]
    diff_2 = [row for row in dataset if row['difficulty'] == 2]

    print(f"Pool Stats: Diff_0: {len(diff_0)}, Diff_1: {len(diff_1)}, Diff_2: {len(diff_2)}")

    # Stratified sampling target: 17 + 17 + 16 = 50
    sample_counts = {0: n//3, 1: n - 2 * (n//3), 2: n//3}

    # Use fixed seed for reproducibility
    random.seed(114514)

    selected_data = []
    selected_data.extend(random.sample(diff_0, sample_counts[0]))
    selected_data.extend(random.sample(diff_1, sample_counts[1]))
    selected_data.extend(random.sample(diff_2, sample_counts[2]))

    # Shuffle the final mix so they aren't ordered by difficulty
    random.shuffle(selected_data)

    output_file = "nl2bash_eval_50.jsonl" if not ofile else ofile

    formatted_lines = []

    for row in selected_data:
        nl_text = row['nl']

        entry = {
            "messages": [
                {"role": "user", "content": nl_text},
            ]
        }
        formatted_lines.append(json.dumps(entry))

    with open(output_file, "w", encoding="utf-8") as f:
        for line in formatted_lines:
            f.write(line + "\n")

    print(f"\nSuccessfully generated file: {output_file}")
    print("-" * 30)
    print("Eval Set Composition:")
    final_diffs = [row['difficulty'] for row in selected_data]
    print(f"- Difficulty 0: {final_diffs.count(0)} records")
    print(f"- Difficulty 1: {final_diffs.count(1)} records")
    print(f"- Difficulty 2: {final_diffs.count(2)} records")
    print("-" * 30)


if __name__ == "__main__":
    generate_finetune_data()
    generate_validation_data()
    generate_eval_data()