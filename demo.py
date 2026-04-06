import random
import re

import numpy as np
import torch
import torch.cuda
import fire
from transformers import BitsAndBytesConfig
import nvtx

from promptcache.model import CodeLlama, Qwen

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


def main(enable_cache=True):
    enable_cache = normalize_bool(enable_cache)

    enable_cpu_inference = False
    disable_prompt_cache = not enable_cache

    lm_for_cache = Qwen("/data/zy/models/Qwen/Qwen3/Qwen3-8B-Base",
                        device_map="cuda:2",
                        **build_quantization_kwargs(True))

    lm = lm_for_cache

    if enable_cpu_inference:
        lm = Qwen("/data/zy/models/Qwen/Qwen3/Qwen3-8B-Base",
                  device_map=None)

    cache_engine = CacheEngine(5000, lm_for_cache, target_device='cpu' if enable_cpu_inference else None)
    gen_engine = GenerationEngine(lm)

    preproc = [
        lm.get_formatter()
    ]

    cache_engine.add_schema(read_file("./examples/mcp_function_call.xml", preproc), max_tokens=800)

    parameter = GenerationParameters(
        temperature=0.0,
        repetition_penalty=1.0,
        top_p=1.0,
        top_k=-1,
        max_new_tokens=512,
        stop_token_ids=lm.stop_token_ids,
        stop_str=lm.stop_str + ["(END_JSON)", "<END_JSON>"]
    )

    # 工具列表，从全量逐步减少到只剩 <weather/>
    all_tools = ["stock", "translate", "calculator", "wiki", "map_search"]
    # 每轮从末尾删一个，直到只剩 <weather/>
    # round 0: all 6 tools, round 1: remove map_search, ..., round 5: only weather
    tool_sets = []
    remaining = list(all_tools)  # stock, translate, calculator, wiki, map_search
    for i in range(len(all_tools) + 1):
        tool_sets.append(["weather"] + remaining[:])
        if remaining:
            remaining.pop()  # 每次从末尾删一个

    ttft_results = {}  # 记录每轮首字 token 时间

    for round_idx, tools in enumerate(tool_sets):
        tools_xml = "\n        ".join(f"<{t}/>" for t in tools)
        prompt_text = f"""
        <prompt schema='mcp-function-call'>
        {tools_xml}
        <user>
            check the weather of Beijing.
        </user>
        </prompt>
        """

        prompt = Prompt(prompt_text, preproc)

        label = f"round_{round_idx}_tools={'_'.join(tools)}"
        print(f"\n{'='*60}")
        print(f"[Round {round_idx}] Tools: {tools}")

        # ── NVTX region 开始：从 process 到首字 token ──
        with nvtx.annotate(f"ttft_{label}", color="green"):

            with nvtx.annotate("cache_engine_process", color="blue"):
                token_ids, position_ids, cache_time, cache = cache_engine.process(
                    prompt,
                    no_cache=disable_prompt_cache,
                    return_full_position_ids=lm.use_full_position_ids
                )

            # 用 CUDA event 精确测量从 generate 调用到首字 token 的时间
            start_event = torch.cuda.Event(enable_timing=True)
            first_token_event = torch.cuda.Event(enable_timing=True)

            output_stream = gen_engine.generate(
                token_ids, position_ids, parameter, cache,
                stream_interval=2,
                use_full_position_ids=lm.use_full_position_ids
            )

            print(f"Assistant: ", end="", flush=True)

            resp = ""
            pre = 0
            first_token_time = None

            with nvtx.annotate("generation_loop", color="yellow"):
                start_event.record()

                for step, outputs in enumerate(output_stream):
                    output_text = outputs.new_text
                    now = len(output_text)

                    if step == 0:
                        # 首字 token 到达
                        first_token_event.record()
                        torch.cuda.synchronize()
                        first_token_time = start_event.elapsed_time(first_token_event)  # ms

                    if now > pre:
                        tt = output_text[pre:now]
                        resp += tt
                        print(tt, end="", flush=True)
                        pre = now

        print("", flush=True)

        resp = resp.replace("(END_JSON)", "").replace("<END_JSON>", "").rstrip()

        ttft_ms = first_token_time if first_token_time is not None else float('nan')
        ttft_results[round_idx] = {
            "tools": tools,
            "cache_time_ms": cache_time * 1000 if cache_time is not None else None,
            "ttft_ms": ttft_ms,
        }

        print(f"\n[Round {round_idx}] cache_time={cache_time:.4f}s | TTFT={ttft_ms:.2f}ms")
        print(f"[Round {round_idx}] Response: {resp[:100]}...")

        prompt_text += f"<assistant>{resp}</assistant>"

    # 汇总输出
    print(f"\n{'='*60}")
    print("TTFT Summary:")
    print(f"{'Round':<8} {'Tools':<50} {'CacheTime(ms)':<16} {'TTFT(ms)':<12}")
    for round_idx, result in ttft_results.items():
        tools_str = str(result['tools'])
        ct = f"{result['cache_time_ms']:.2f}" if result['cache_time_ms'] is not None else "N/A"
        print(f"{round_idx:<8} {tools_str:<50} {ct:<16} {result['ttft_ms']:<12.2f}")


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