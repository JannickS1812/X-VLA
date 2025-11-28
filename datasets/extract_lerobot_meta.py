import json
import argparse
from pathlib import Path

def load_jsonl(path):
    """Helper to load a .jsonl file as a list of dictionaries."""
    data = []
    if path.exists():
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
    return data

def extract_dataset_info(dataset_path, output_file, dataset_name_override):
    dataset_root = Path(dataset_path)
    
    # 1. Load info.json (The second file you provided)
    info_path = dataset_root / "meta/info.json"
    if not info_path.exists():
        print(f"Error: Could not find info.json at {info_path}")
        return

    with open(info_path, 'r', encoding='utf-8') as f:
        info_data = json.load(f)

    # 2. Load Episode Metadata
    # LeRobot stores per-episode length and task_index in meta/episodes.jsonl
    episodes_path = dataset_root / "meta/episodes.jsonl"
    if not episodes_path.exists():
        print(f"Error: Could not find meta/episodes.jsonl at {episodes_path}")
        print("Note: info.json only contains global stats. The 'meta' folder is required for per-episode details.")
        return
    
    episodes_list = load_jsonl(episodes_path)

    # 3. Load Task Metadata (Optional)
    # Maps task_index to a description strings
    tasks_path = dataset_root / "meta/tasks.jsonl"
    task_map = {}
    
    if tasks_path.exists():
        tasks_data = load_jsonl(tasks_path)
        for t in tasks_data:
            # Map index to task_description
            if 'task_index' in t and 'task' in t:
                task_map[t['task_index']] = t['task']
    else:
        print("Warning: meta/tasks.jsonl not found. Using generic task names.")

    # 4. Construct the Output Structure
    output_data = {
        "dataset_name": dataset_name_override if dataset_name_override else "LeRobot_Dataset",
        "datalist": []
    }

    print(f"Processing {len(episodes_list)} episodes...")

    for i, ep in enumerate(episodes_list):
        # Retrieve task description based on index
        task_idx = ep.get("task_index", 0)
        task_desc = task_map.get(task_idx, f"Task {task_idx}")

        # Construct the entry
        entry = {
            "top_path": str(dataset_root.absolute()), # Or specific subdirectory if preferred
            "episode_index": ep.get("episode_index", i),
            "tasks": [
                task_desc
            ],
            "length": ep.get("length", 0)
        }
        output_data["datalist"].append(entry)

    # 5. Save to file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4)
    
    print(f"Successfully successfully extracted {len(output_data['datalist'])} episodes to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert LeRobot dataset metadata to custom JSON format.")
    parser.add_argument("dataset_path", type=str, help="Path to the root of the LeRobot dataset (containing info.json)")
    parser.add_argument("--output", type=str, default="output_datalist.json", help="Name of the output file")
    parser.add_argument("--name", type=str, default="AGIBOT", help="Name of the dataset")

    args = parser.parse_args()
    
    extract_dataset_info(args.dataset_path, args.output, args.name)