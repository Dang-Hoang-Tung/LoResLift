import json
import os
import re
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Tuple

import jsonlines
from tqdm import tqdm

from prompts import J2K_FIXUP_TEMPLATE, J2K_TRANSLATION_TEMPLATE
from utils import clean_answer, generate, read_jsonl, run_generic_pipeline, write_jsonl


# --------------------------------
# Direct
# --------------------------------

def run_direct(kotlin_prompt_dict: Dict[str, str], model, tokenizer, output_eval_file: str):
    """Generate Kotlin completions directly from the Kotlin prompts."""
    os.makedirs(os.path.dirname(output_eval_file), exist_ok=True)

    kotlin_outputs = []
    task_ids = list(kotlin_prompt_dict)

    for task_id in tqdm(task_ids, desc="Generating", leave=False):
        print(f"=== {task_id} ===")
        prompt = kotlin_prompt_dict[task_id]
        print("Prompt:\n", prompt)
        raw = generate(prompt, model, tokenizer, device="cuda")
        cleaned = clean_answer(raw, "kotlin")
        print("Cleaned output:\n", cleaned)
        kotlin_outputs.append({"task_id": task_id, "completion": cleaned, "language": "kotlin"})

    with jsonlines.open(output_eval_file, mode="w") as writer:
        writer.write_all(kotlin_outputs)

    print(f"Wrote {len(kotlin_outputs)} completions to {output_eval_file}")


# --------------------------------
# Shared pivot helpers
# --------------------------------

def load_failed_samples(results_direct_jsonl_path: str, problem_dict: Dict) -> Tuple[Dict, List]:
    """Return (failed_problem_dict, failed_kotlin_samples) based on DIRECT results."""
    results = list(read_jsonl(results_direct_jsonl_path))
    results_dict = {item["task_id"]: item for item in results}

    failed_problem_dict = {
        key: problem_dict[key]
        for key in results_dict
        if not results_dict[key]["passed"]
    }
    failed_kotlin_samples = list(failed_problem_dict.values())

    print(f"Failed problems: {list(failed_problem_dict.keys())}")
    return failed_problem_dict, failed_kotlin_samples


def run_pivot_java_gen(
    java_prompt_dict: Dict[str, str],
    failed_problem_dict: Dict,
    model,
    tokenizer,
    output_java_file: str,
):
    """Generate Java solutions for the failed Kotlin problems."""
    os.makedirs(os.path.dirname(output_java_file), exist_ok=True)

    relevant_java_prompt_dict = {
        task_id: content
        for task_id, content in java_prompt_dict.items()
        if task_id in failed_problem_dict
    }

    java_gen_outputs = []
    for task_id in tqdm(relevant_java_prompt_dict, desc="Generating Java", leave=False):
        print(f"=== {task_id} ===")
        prompt = relevant_java_prompt_dict[task_id]
        print("Prompt:\n", prompt)
        raw = generate(prompt, model, tokenizer, device="cuda")
        cleaned = clean_answer(raw, "java")
        print("Cleaned output:\n", cleaned)
        java_gen_outputs.append({"task_id": task_id, "completion": cleaned, "language": "java"})

    with jsonlines.open(output_java_file, mode="w") as writer:
        writer.write_all(java_gen_outputs)

    print(f"Wrote {len(java_gen_outputs)} Java completions to {output_java_file}")


# --------------------------------
# Pivot LLM
# --------------------------------

def run_pivot_llm(
    failed_kotlin_samples: List[Any],
    output_java_file: str,
    kotlin_signatures: Dict[str, str],
    tokenizer,
    model,
    output_dir: str,
    output_eval_file: str,
):
    """Translate LLM-generated Java solutions back to Kotlin using an LLM."""
    java_generated_list = list(read_jsonl(output_java_file))
    java_generated_dict = {item["task_id"]: item for item in java_generated_list}

    def get_extra_data(sid: str) -> Dict:
        return {
            "java_src_code": java_generated_dict[sid]["completion"],
            "kotlin_signature": kotlin_signatures[sid],
        }

    run_generic_pipeline(
        out_dir=output_dir,
        relevant_kotlin_samples=failed_kotlin_samples,
        get_extra_data=get_extra_data,
        get_output_file_name=lambda sid: f"{sid.split('/')[-1]}_back_translated.kt",
        out_jsonl=output_eval_file,
        tokenizer=tokenizer,
        model=model,
        prompt_template=J2K_TRANSLATION_TEMPLATE,
    )


# --------------------------------
# Pivot RB
# --------------------------------

def export_for_rb_translation(output_java_file: str, output_pivot_dir: str):
    """Write each Java completion as an individual .java file for manual RB translation."""
    translate_dir = os.path.join(output_pivot_dir, "to_translate")
    os.makedirs(translate_dir, exist_ok=True)

    java_outputs = list(read_jsonl(output_java_file))
    for item in java_outputs:
        file_id = item["task_id"].split("/")[-1]
        path = os.path.join(translate_dir, f"{file_id}.java")
        with open(path, "w", encoding="utf-8") as f:
            f.write(item["completion"])

    print(f"Exported {len(java_outputs)} .java files to: {translate_dir}")


_WRAP_START_RE = re.compile(r'^\s*internal\s+(class|object)\b[^{]*\{\s*$')


def _find_wrap_bounds(lines: List[str]) -> Tuple[int, int]:
    """Return (start_idx, end_idx) for an 'internal class/object { ... }' wrapper."""
    start = None
    for i, line in enumerate(lines):
        if _WRAP_START_RE.match(line):
            start = i
            break
    if start is None:
        return None, None

    brace_count = 1
    for j in range(start + 1, len(lines)):
        brace_count += lines[j].count("{")
        brace_count -= lines[j].count("}")
        if brace_count == 0:
            return start, j
    return None, None


def run_pivot_rb_cleanup(
    output_pivot_dir: str,
    output_dir: str,
    pipeline_type: str,
    output_file: str,
    output_eval_file: str,
):
    """
    Unwrap IntelliJ-translated .kt files from their 'internal class/object' wrapper
    and write results to a JSONL file.
    """
    translated_dir = Path(output_pivot_dir) / "translated"
    translated_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for p in translated_dir.glob("*.kt"):
        if not p.is_file():
            continue

        src = p.read_text(encoding="utf-8")
        lines = src.splitlines(keepends=True)
        s, e = _find_wrap_bounds(lines)

        if s is not None and e is not None:
            inner = "".join(lines[s + 1:e])
            output = textwrap.dedent(inner)
        else:
            output = src

        stem = p.name.replace(".kt", "")
        out_path = Path(output_dir) / f"{stem}_back_translated.kt"
        out_path.write_text(output, encoding="utf-8")

        results.append({
            "task_id": f"HumanEval_kotlin/{stem}",
            "completion": output,
            "language": "kotlin",
        })
        print(f"Processed {p.name}")

    out_jsonl = output_file if pipeline_type == "PIVOT_RB_LLM" else output_eval_file
    write_jsonl(results, out_jsonl)
    print(f"Wrote {len(results)} results to: {out_jsonl}")


# --------------------------------
# Pivot RB + LLM fixup
# --------------------------------

def run_pivot_rb_llm_fixup(
    failed_kotlin_samples: List[Any],
    kotlin_signatures: Dict[str, str],
    output_dir: str,
    tokenizer,
    model,
    output_eval_file: str,
):
    """LLM fixup pass on rule-based translated Kotlin files."""
    def get_extra_data(sid: str) -> Dict:
        sid_short = sid.split("/")[-1]
        content = open(os.path.join(output_dir, f"{sid_short}_back_translated.kt"), encoding="utf-8").read()
        return {
            "kotlin_src_code": content,
            "kotlin_signature": kotlin_signatures[sid],
        }

    run_generic_pipeline(
        out_dir=output_dir,
        relevant_kotlin_samples=failed_kotlin_samples,
        get_extra_data=get_extra_data,
        get_output_file_name=lambda sid: f"{sid.split('/')[-1]}_fixup.kt",
        out_jsonl=output_eval_file,
        tokenizer=tokenizer,
        model=model,
        prompt_template=J2K_FIXUP_TEMPLATE,
    )


# --------------------------------
# Evaluate
# --------------------------------

def run_evaluate(output_eval_file: str, problem_dict: Dict, results_file: str):
    """Run functional correctness evaluation and print pass rate."""
    from mxeval.evaluation import evaluate_functional_correctness

    final_output_list = list(read_jsonl(output_eval_file))
    final_output_dict = {item["task_id"]: item for item in final_output_list}

    relevant_problem_dict = {
        task_id: value
        for task_id, value in problem_dict.items()
        if task_id in final_output_dict
    }

    evaluate_functional_correctness(
        sample_file=output_eval_file,
        k=[1],
        n_workers=8,
        timeout=15,
        problem_file=relevant_problem_dict,
    )

    total, correct = 0, 0
    with open(results_file) as f:
        for line in f:
            res = json.loads(line)
            print(res)
            total += 1
            correct += res.get("passed", 0)

    print(f"Pass rate: {correct}/{total} = {correct / total:.2%}")
