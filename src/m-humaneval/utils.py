import json
import os
import re
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import torch
from transformers import GenerationConfig, StoppingCriteria, StoppingCriteriaList


# --------------------------------
# Stopping criteria
# --------------------------------

class StopOnSecondFence(StoppingCriteria):
    def __init__(self, tokenizer, start_len: int, max_check_tokens: int = 2048):
        self.tokenizer = tokenizer
        self.start_len = start_len
        self.max_check_tokens = max_check_tokens

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        tail_ids = input_ids[0, self.start_len:].tolist()
        if not tail_ids:
            return False
        tail_ids = tail_ids[-self.max_check_tokens:]
        text = self.tokenizer.decode(tail_ids, skip_special_tokens=True)
        return text.count("```") >= 2


class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops, tokenizer):
        (StoppingCriteria.__init__(self),)
        self.stops = rf"{stops}"
        self.tokenizer = tokenizer

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        last_three_tokens = [int(x) for x in input_ids.data[0][-3:]]
        decoded_last_three_tokens = self.tokenizer.decode(last_three_tokens)
        return bool(re.search(self.stops, decoded_last_three_tokens))


# --------------------------------
# Generation
# --------------------------------

def generate(problem: str, model, tokenizer, device) -> str:
    """Completion-style generation, stops at closing brace."""
    criterion = StoppingCriteriaSub(stops="\n}\n", tokenizer=tokenizer)
    stopping_criteria = StoppingCriteriaList([criterion])

    inputs = tokenizer.encode(problem, return_tensors="pt").to(device)

    outputs = model.generate(
        inputs,
        max_new_tokens=2048,
        min_new_tokens=128,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=False,
        temperature=0.0,
        top_p=1.0,
        num_beams=1,
        stopping_criteria=stopping_criteria,
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def generate_once(tokenizer, model, messages: List[Dict[str, str]], max_new_tokens: int = 3072) -> str:
    """Chat/instruction-style generation, stops after the second ``` fence."""
    if getattr(tokenizer, "chat_template", None):
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt = "\n".join(m["content"] for m in messages)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    start_len = inputs["input_ids"].shape[1]
    stops = StoppingCriteriaList([StopOnSecondFence(tokenizer, start_len)])

    gen_cfg = GenerationConfig(
        do_sample=False,
        temperature=0.0,
        top_p=1.0,
        max_new_tokens=max_new_tokens,
        eos_token_id=[tokenizer.eos_token_id],
    )

    outputs = model.generate(**inputs, generation_config=gen_cfg, stopping_criteria=stops)
    gen_only = outputs[0, start_len:]
    return tokenizer.decode(gen_only, skip_special_tokens=True).strip()


def clean_answer(code: str, lang: str) -> str:
    code = re.sub(r"/\*.*?\*/", "", code, flags=re.DOTALL)
    code = re.sub(r"//.*", "", code)

    lines = code.strip().split("\n")
    function_prefix = "fun " if lang == "kotlin" else "public "

    start_idx = -1
    for i, line in enumerate(lines):
        if line.strip().startswith(function_prefix):
            start_idx = i
            break

    if start_idx == -1:
        return code

    end_idx = len(lines)
    for j in range(start_idx, len(lines)):
        if lines[j].startswith("}"):
            end_idx = j + 1
            break

    return "\n".join(lines[start_idx:end_idx])


# --------------------------------
# Fenced block extraction
# --------------------------------

FENCE_RE = re.compile(r"(?s)```([A-Za-z0-9#+\-.]*)\s*\n(.*?)\n```")


def extract_single_fenced_code_block(text: str, expected_language: Optional[str] = None) -> Tuple[Optional[str], str]:
    """Return (language, code). Raises if not exactly 1 block found after filtering."""
    matches = FENCE_RE.findall(text)
    if expected_language:
        matches = [(lang, body) for (lang, body) in matches if lang.lower() == expected_language.lower()]
    if len(matches) != 1:
        raise ValueError(f"Found {len(matches)} fenced code blocks; expected exactly 1.")
    lang, body = matches[0]
    return (lang or None), body.strip()


# --------------------------------
# JSONL I/O
# --------------------------------

def read_jsonl(path: str) -> Iterable[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception as e:
                raise ValueError(f"Failed to parse JSON on line {line_no} of {path}: {e}")


def write_jsonl(records: Iterable[Dict], path: str):
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# --------------------------------
# Generic pipeline runner
# --------------------------------

def run_generic_pipeline(
    out_dir: str,
    relevant_kotlin_samples: List[Any],
    get_extra_data: Optional[Callable[[str], Dict[str, Any]]],
    get_output_file_name: Callable[[str], str],
    out_jsonl: Optional[str],
    tokenizer,
    model,
    prompt_template: str,
):
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    results = []
    error_count = 0
    success_count = 0

    for idx, kotlin_sample in enumerate(relevant_kotlin_samples, 1):
        sid = kotlin_sample.get("task_id")
        current_data = {"fn_description": kotlin_sample.get("description", "").strip()}

        if get_extra_data is not None:
            current_data.update(get_extra_data(sid))

        messages = [{"role": "user", "content": prompt_template.format(**current_data)}]
        print(messages[0]["content"])
        raw = generate_once(tokenizer, model, messages)
        print(raw)

        status = "ok"
        code = ""
        try:
            _, code = extract_single_fenced_code_block(raw, expected_language="kotlin")
        except Exception as e:
            status = "error"
            print(str(e))
            error_count += 1

        if status == "ok":
            fname = get_output_file_name(sid)
            path = os.path.join(out_dir, fname)
            with open(path, "w", encoding="utf-8") as f:
                f.write(code)
            results.append({"task_id": sid, "completion": code, "language": "kotlin"})
            success_count += 1

        print(f"[{idx}/{len(relevant_kotlin_samples)}] id={sid} status={status}")

    print(f"Errors: {error_count}/{len(relevant_kotlin_samples)}")
    print(f"Success: {success_count}/{len(relevant_kotlin_samples)}")

    if out_jsonl:
        write_jsonl(results, out_jsonl)
        print(f"Done. Wrote results to: {out_jsonl}")
