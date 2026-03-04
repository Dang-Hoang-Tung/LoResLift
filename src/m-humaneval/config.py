from enum import Enum
from types import SimpleNamespace
import os


class OutputPath(str, Enum):
    DIRECT = "results-direct"
    PIVOT_LLM = "results-pivot-llm"
    PIVOT_RB = "results-pivot-rb"
    PIVOT_RB_LLM = "results-pivot-rb-llm"


def make_paths(model_id: str, pipeline_type: str, project_dir: str) -> SimpleNamespace:
    model_nickname = model_id.replace("/", "__")
    output_group_dir = os.path.join(project_dir, "results", model_nickname)
    output_dir = os.path.join(output_group_dir, OutputPath[pipeline_type])
    output_pivot_dir = os.path.join(output_group_dir, "results-pivot")
    return SimpleNamespace(
        output_group_dir=output_group_dir,
        output_dir=output_dir,
        output_file=os.path.join(output_dir, "output.jsonl"),
        output_eval_file=os.path.join(output_dir, "output_eval.jsonl"),
        results_file=os.path.join(output_dir, "output_eval.jsonl_results.jsonl"),
        output_pivot_dir=output_pivot_dir,
        output_java_file=os.path.join(output_pivot_dir, "java_output.jsonl"),
        java_prompts_dir=os.path.join(project_dir, "Prompts", "translated"),
    )
