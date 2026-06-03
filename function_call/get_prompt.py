import random
import re
import json
from pathlib import Path

try:
    import fire
except Exception:
    fire = None

try:
    import numpy as np
except Exception:
    np = None

try:
    import torch
except Exception:
    torch = None

try:
    from promptcache.model import Llama2, Falcon, Mpt, CodeLlama
    from promptcache import Prompt, CompactSpaces, read_file, CacheEngine, \
        GenerationEngine, GenerationParameters, llama2_template
except Exception:
    Llama2 = Falcon = Mpt = CodeLlama = None
    Prompt = CompactSpaces = read_file = CacheEngine = None
    GenerationEngine = GenerationParameters = llama2_template = None
 

def escape_tags(input_str):
    pattern = r'<(?P<content>.*?)>'

    def repl(match):
        return '(' + match.group("content").capitalize() + ')'

    return re.sub(pattern, repl, input_str)

def extract_tool_names_from_system(system_content: str):
    if not system_content:
        return []
    blocks = re.findall(r"<tools>\s*(.*?)\s*</tools>", system_content, flags=re.DOTALL)
    if not blocks:
        return []
    tools_block = max(blocks, key=lambda b: b.count('"name"'))
    return re.findall(r'"name"\s*:\s*"([^"]+)"', tools_block)


def extract_selected_tool_name(messages, tool_names):
    assistant_contents = [
        m.get("content", "")
        for m in messages
        if m.get("role") == "assistant"
    ]
    assistant_text = "\n".join(assistant_contents)

    m = re.search(r"<tool_call>[\s\S]*?\"name\"\s*:\s*\"([^\"]+)\"", assistant_text)
    if m:
        name = m.group(1)
        if not tool_names or name in tool_names:
            return name

    for name in tool_names:
        if re.search(rf"\b{re.escape(name)}\b", assistant_text):
            return name

    if len(tool_names) == 1:
        return tool_names[0]
    return tool_names[0] if tool_names else "calculate_tip"


def _dedupe_preserve_order(items):
    seen = set()
    out = []
    for x in items:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def build_prompt_text(tool_names, user_query: str):
    tool_names = [t.strip() for t in (tool_names or []) if str(t).strip()]
    tool_names = _dedupe_preserve_order(tool_names)
    if not tool_names:
        tool_names = ["calculate_tip"]

    tool_tags = "\n".join([f"        <{name}/>" for name in tool_names])
    user_query = escape_tags((user_query or "").strip())
    return f"""
        <prompt schema='function-call'>
{tool_tags}
        <function-call/>
        <user>
            {user_query}
        </user>
        </prompt>
        """


def main(
    enable_cache=True,
    max_ctx_length=2048,
    schema_max_tokens=512,
    max_new_tokens=256,
    dataset_path="/data/xingye/data/glaive-function-calling-v2/dataset_qwen.json",
    output_path="./outputs/dataset_qwen_results.jsonl",
    limit=-1,
    shuffle=False,
    print_every=50,
    dry_run=False,
    write_prompt=False,
    only_called=False,
):
    
    enable_cpu_inference = False

    disable_prompt_cache = not enable_cache

    dataset_path = str(dataset_path)
    output_path = str(output_path)
    data = json.loads(Path(dataset_path).read_text(encoding="utf-8"))
    if shuffle:
        random.shuffle(data)
    if int(limit) > 0:
        data = data[: int(limit)]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        if dry_run:
            for i, item in enumerate(data):
                messages = item.get("messages", [])
                system_content = next((m.get("content", "") for m in messages if m.get("role") == "system"), "")
                user_query = next((m.get("content", "") for m in messages if m.get("role") == "user"), "")

                tool_names = extract_tool_names_from_system(system_content)
                function_name = extract_selected_tool_name(messages, tool_names)
                prompt_tool_names = [function_name] if only_called else tool_names
                prompt_text = build_prompt_text(prompt_tool_names, user_query)

                record = {
                    "idx": i,
                    "function_name": function_name,
                    "tool_names": tool_names,
                    "user": user_query,
                }
                if write_prompt:
                    record["prompt"] = prompt_text
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                if int(print_every) > 0 and (i + 1) % int(print_every) == 0:
                    print(f"[{i + 1}/{len(data)}] last_function={function_name}")
            return

    if Prompt is None or CodeLlama is None or CacheEngine is None:
        raise ImportError("promptcache/torch not available; run with --dry_run=True or install dependencies")

    lm_for_cache = CodeLlama(
        "codellama/CodeLlama-7b-Instruct-hf",
        load_in_8bit=True,
        device_map="auto",
    )

    lm = lm_for_cache

    if enable_cpu_inference:
        lm = CodeLlama(
            "codellama/CodeLlama-7b-Instruct-hf",
            load_in_8bit=False,
            device_map=None,
        )

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
        read_file("./schema/function.xml", preproc),
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

    with open(output_path, "w", encoding="utf-8") as f:
        for i, item in enumerate(data):
            messages = item.get("messages", [])
            system_content = next((m.get("content", "") for m in messages if m.get("role") == "system"), "")
            user_query = next((m.get("content", "") for m in messages if m.get("role") == "user"), "")

            tool_names = extract_tool_names_from_system(system_content)
            function_name = extract_selected_tool_name(messages, tool_names)

            prompt_tool_names = [function_name] if only_called else tool_names
            prompt_text = build_prompt_text(prompt_tool_names, user_query)
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

            resp = ""
            pre = 0
            for outputs in output_stream:
                output_text = outputs.new_text.strip().split(" ")
                now = len(output_text) - 1
                if now > pre:
                    tt = " ".join(output_text[pre:now])
                    resp += tt + " "
                    pre = now
            tt = " ".join(output_text[pre:])
            resp += tt
            resp = resp.strip()

            record = {
                "idx": i,
                "function_name": function_name,
                "tool_names": tool_names,
                "user": user_query,
                "assistant": resp,
            }
            if write_prompt:
                record["prompt"] = prompt_text
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            if int(print_every) > 0 and (i + 1) % int(print_every) == 0:
                print(f"[{i + 1}/{len(data)}] last_function={function_name}")


def seed_everything(seed):
    if torch is None or np is None:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    seed_everything(42)
    if fire is not None:
        fire.Fire(main)
    else:
        import argparse

        def str2bool(v):
            if isinstance(v, bool):
                return v
            v = str(v).lower()
            if v in {"1", "true", "t", "yes", "y"}:
                return True
            if v in {"0", "false", "f", "no", "n"}:
                return False
            raise argparse.ArgumentTypeError(f"invalid bool: {v}")

        p = argparse.ArgumentParser()
        p.add_argument("--enable_cache", type=str2bool, default=True)
        p.add_argument("--max_ctx_length", type=int, default=2048)
        p.add_argument("--schema_max_tokens", type=int, default=512)
        p.add_argument("--max_new_tokens", type=int, default=256)
        p.add_argument("--dataset_path", type=str, default="/data/xingye/data/glaive-function-calling-v2/dataset_qwen.json")
        p.add_argument("--output_path", type=str, default="./outputs/dataset_qwen_results.jsonl")
        p.add_argument("--limit", type=int, default=-1)
        p.add_argument("--shuffle", type=str2bool, default=False)
        p.add_argument("--print_every", type=int, default=50)
        p.add_argument("--dry_run", type=str2bool, default=False)
        p.add_argument("--write_prompt", type=str2bool, default=False)
        p.add_argument("--only_called", type=str2bool, default=False)
        args = vars(p.parse_args())
        main(**args)
