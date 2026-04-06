import random
import re

import numpy as np
import torch.cuda
import fire

from promptcache.model import Llama2, Falcon, Mpt, CodeLlama

from promptcache import Prompt, CompactSpaces, read_file, CacheEngine, \
    GenerationEngine, GenerationParameters, llama2_template
 

def escape_tags(input_str):
    pattern = r'<(?P<content>.*?)>'

    def repl(match):
        return '(' + match.group("content").capitalize() + ')'

    return re.sub(pattern, repl, input_str)


def main(enable_cache=True):
    
    enable_cpu_inference = False
    disable_prompt_cache = not enable_cache

    lm_for_cache = CodeLlama("codellama/CodeLlama-7b-Instruct-hf",
                             load_in_8bit=True,
                             device_map="cuda:2")

    lm = lm_for_cache

    if enable_cpu_inference:
        lm = CodeLlama("codellama/CodeLlama-7b-Instruct-hf",
                       load_in_8bit=False,
                       device_map=None)

    cache_engine = CacheEngine(5000, lm_for_cache, target_device='cpu' if enable_cpu_inference else None)
    gen_engine = GenerationEngine(lm)

    preproc = [
        lm.get_formatter()
    ]

    cache_engine.add_schema(read_file("./examples/mcp_function_call.xml", preproc), max_tokens=800)

    parameter = GenerationParameters(
        temperature=1.0,
        repetition_penalty=1.0,
        top_p=0.95,
        top_k=-1,
        max_new_tokens=512,
        stop_token_ids=lm.stop_token_ids,
        stop_str=lm.stop_str
    )

    # 按顺序逐个删除的标签列表（从后往前删）
    tags_to_remove = ["<map_search/>", "<wiki/>", "<calculator/>", "<translate/>", "<stock/>"]

    # 初始包含全部6个标签
    current_tags = ["<weather/>", "<stock/>", "<translate/>", "<calculator/>", "<wiki/>", "<map_search/>"]

    def build_prompt(tags):
        tags_str = "\n        ".join(tags)
        return f"""
        <prompt schema='mcp-function-call'>
        {tags_str}
        <user>
            Check the weather of Beijing.
        </user>
        </prompt>
        """

    def run(prompt_text: str):
        prompt = Prompt(prompt_text, preproc)

        token_ids, position_ids, cache_time, cache = cache_engine.process(
            prompt,
            no_cache=disable_prompt_cache,
            return_full_position_ids=lm.use_full_position_ids
        )

        output_stream = gen_engine.generate(
            token_ids, position_ids, parameter, cache,
            stream_interval=2,
            use_full_position_ids=lm.use_full_position_ids
        )

        print(f"Assistant: ", end="", flush=True)

        resp = ""
        pre = 0
        for outputs in output_stream:
            output_text = outputs.new_text.strip().split(" ")
            now = len(output_text) - 1
            if now > pre:
                tt = " ".join(output_text[pre:now])
                resp += tt + " "
                print(tt, end=" ", flush=True)
                pre = now
        tt = " ".join(output_text[pre:])
        print(tt, flush=True)
        resp += tt
        print("\n")

    # 第1次：包含全部6个标签
    for i, tag_to_remove in enumerate([None] + tags_to_remove):
        if tag_to_remove is not None:
            current_tags.remove(tag_to_remove)

        print(f"\n{'='*50}")
        print(f"Round {i+1}: tags = {current_tags}")
        print(f"{'='*50}")

        prompt_text = build_prompt(current_tags)
        run(prompt_text)

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