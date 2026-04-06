import random
import re

import json
import sys
from pathlib import Path

import numpy as np
import torch.cuda
import fire
from transformers import BitsAndBytesConfig

sys.path.insert(1, str(Path(__file__).resolve().parents[1]))

from promptcache.model import Qwen

from promptcache import Prompt, CompactSpaces, read_file, CacheEngine, \
    GenerationEngine, GenerationParameters, llama2_template
 

def escape_tags(input_str):
    pattern = r'<(?P<content>.*?)>'

    def repl(match):
        return '(' + match.group("content").capitalize() + ')'

    return re.sub(pattern, repl, input_str)


def normalize_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"1", "true", "t", "yes", "y", "on"}:
            return True
        if v in {"0", "false", "f", "no", "n", "off"}:
            return False
    return bool(value)


def build_quantization_kwargs(enable_8bit):
    if normalize_bool(enable_8bit):
        return {"quantization_config": BitsAndBytesConfig(load_in_8bit=True)}
    return {}


def main(
    enable_cache=True,
    max_ctx_length=2048,
    schema_max_tokens=512,
    max_new_tokens=256,
    model_name="/data/zy/models/Qwen/Qwen3/Qwen3-8B-Base",
    dataset_path="/home/xingye/prompt-cache/function_call/outputs/only_called_prompts.jsonl",
    output_path=None,
    start=0,
    limit=None,
    print_every=1,
):
    enable_cache = normalize_bool(enable_cache)
    
    enable_cpu_inference = False

    disable_prompt_cache = not enable_cache


    lm_for_cache = Qwen(str(model_name),
                        device_map="auto",
                        **build_quantization_kwargs(True))

    lm = lm_for_cache

    if enable_cpu_inference:
        lm = Qwen(str(model_name),
                  device_map=None)

    cache_engine = CacheEngine(
        int(max_ctx_length),
        lm_for_cache,
        target_device="cpu" if enable_cpu_inference else None,
    )
    gen_engine = GenerationEngine(lm)

    preproc = [
        CompactSpaces(),
        lm.get_formatter()
    ]

    cache_engine.add_schema(
        read_file(str(Path(__file__).resolve().parent / "schema" / "function.xml"), preproc),
        max_tokens=int(schema_max_tokens),
    )

    parameter = GenerationParameters(
        temperature=1.0,
        repetition_penalty=1.0,
        top_p=0.95,
        top_k=-1,
        max_new_tokens=int(max_new_tokens),
        stop_token_ids=lm.stop_token_ids,
        stop_str=lm.stop_str
    )

    dataset_path = str(dataset_path)
    if output_path is None:
        p = Path(dataset_path)
        output_path = str(p.with_suffix("").with_name(p.stem + "_outputs.json"))

    out_fp = Path(output_path)
    out_fp.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    seen = 0
    with open(dataset_path, "r", encoding="utf-8") as fin, open(out_fp, "w", encoding="utf-8") as fout:
        fout.write("[\n")
        json_first = True
        for line in fin:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if seen < int(start):
                seen += 1
                continue
            if limit is not None and written >= int(limit):
                break
            seen += 1

            prompt_text = item.get("prompt")
            if not isinstance(prompt_text, str) or not prompt_text.strip():
                result = {
                    **item,
                    "error": "missing_or_empty_prompt",
                }
            else:
                try:
                    prompt = Prompt(prompt_text, preproc)

                    token_ids, position_ids, cache_time, cache = cache_engine.process(
                        prompt,
                        no_cache=disable_prompt_cache,
                        return_full_position_ids=lm.use_full_position_ids,
                    )

                    output_stream = gen_engine.generate(
                        token_ids,
                        position_ids,
                        parameter,
                        cache,
                        stream_interval=2,
                        use_full_position_ids=lm.use_full_position_ids,
                    )

                    last_new_text = ""
                    last_out = None
                    prefill = True
                    prefill_latency_ms = 0;
                    for out in output_stream:
                        if prefill:
                            prefill = False
                            prefill_latency_ms = float(out.response_time)
                        last_new_text = out.new_text
                        last_out = out
                    
                    resp = last_new_text.strip()
                    result = {
                        **item,
                        "response": resp,
                        "cache_time_ms": float(cache_time),
                        "prompt_tokens": int(len(token_ids)),
                        "position_ids": int(len(position_ids)),
                        "prefill_latency_ms": prefill_latency_ms,
                    }
                    if last_out is not None:
                        result["total_inference_ms"] = float(last_out.elapsed_time)
                except Exception as e:
                    result = {
                        **item,
                        "error": f"{type(e).__name__}: {e}",
                    }

            if not json_first:
                fout.write(",\n")
            fout.write(json.dumps(result, ensure_ascii=False))
            fout.flush()
            json_first = False
            written += 1

            if int(print_every) > 0 and written % int(print_every) == 0:
                idx = result.get("idx", None)
                status = "ok" if "response" in result else "error"
                print(f"[{written}] idx={idx} {status}", flush=True)

        fout.write("\n]\n")
    print(f"Saved results to: {out_fp}", flush=True)


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    seed_everything(42)
    fire.Fire(main)
