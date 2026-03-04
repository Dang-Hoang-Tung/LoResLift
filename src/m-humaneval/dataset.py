import json
from typing import Dict, List, Tuple


def load_kotlin_dataset(jsonl_path: str) -> Tuple[List, Dict, Dict, Dict]:
    """
    Load a HumanEval Kotlin JSONL file.

    Returns:
        problem_list: flat list of all records
        problem_dict: {task_id -> record}  (prompt stripped of its final signature line)
        kotlin_prompt_dict: {task_id -> full original prompt}
        kotlin_signatures: {task_id -> last line of the prompt (the function signature)}
    """
    with open(jsonl_path, "r", encoding="utf-8") as f:
        problem_list = [json.loads(line) for line in f]

    problem_dict = {item["task_id"]: item for item in problem_list}

    kotlin_prompt_dict = {
        task_id: item["prompt"]
        for task_id, item in problem_dict.items()
    }

    kotlin_signatures = {}
    for key, item in problem_dict.items():
        lines = item["prompt"].strip().split("\n")
        kotlin_signatures[key] = lines[-1]
        problem_dict[key]["prompt"] = "\n".join(lines[:-1])

    return problem_list, problem_dict, kotlin_prompt_dict, kotlin_signatures


def load_java_dataset(jsonl_path: str) -> Dict[str, str]:
    """
    Load a HumanEval Java JSONL file, returning prompts keyed by the
    equivalent Kotlin task_id (HumanEval_kotlin/N) for easy cross-reference.
    """
    with open(jsonl_path, "r", encoding="utf-8") as f:
        java_list = [json.loads(line) for line in f]

    java_prompt_dict = {}
    for item in java_list:
        n = item["task_id"].split("/")[-1]
        kotlin_task_id = f"HumanEval_kotlin/{n}"
        java_prompt_dict[kotlin_task_id] = item["prompt"]

    return java_prompt_dict
