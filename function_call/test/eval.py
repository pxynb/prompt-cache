import concurrent.futures
import importlib.util
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from promptcache import CacheEngine, CompactSpaces, GenerationEngine, GenerationParameters, Prompt, read_file
from promptcache.model import Qwen

SCENARIOS = [
    {"name": "predictor_gpu_cache", "use_predictor": True, "disable_cache": False, "group": "实验目标"},
    {"name": "full_gpu_no_cache", "use_predictor": False, "disable_cache": True, "group": "对比基线"},
    {"name": "full_gpu_cache", "use_predictor": False, "disable_cache": False, "group": "消融"},
    {"name": "predictor_gpu_no_cache", "use_predictor": True, "disable_cache": True, "group": "消融"},
]


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
        try:
            from transformers import BitsAndBytesConfig
        except Exception:
            return {}
        return {"quantization_config": BitsAndBytesConfig(load_in_8bit=True)}
    return {}


def extract_tool_name_relaxed(content: str, known_tools: Optional[Set[str]] = None) -> Optional[str]:
    if not isinstance(content, str):
        return None
    content = content.strip()
    if not content:
        return None
    candidates = []
    if "<tool_call>" in content and "</tool_call>" in content:
        block = content.split("<tool_call>", 1)[1].split("</tool_call>", 1)[0]
        m = re.search(r'"name"\s*:\s*"([a-zA-Z0-9_]+)"', block)
        if m:
            candidates.append(m.group(1))
        m = re.search(r'name\s*:\s*"?([a-zA-Z0-9_]+)"?', block)
        if m:
            candidates.append(m.group(1))
    for m in re.finditer(r'"name"\s*:\s*"([a-zA-Z0-9_]+)"', content):
        candidates.append(m.group(1))
    for m in re.finditer(r'name\s*:\s*"?([a-zA-Z0-9_]+)"?', content):
        candidates.append(m.group(1))
    for m in re.finditer(r"<([a-zA-Z0-9_]+)\s*/>", content):
        candidates.append(m.group(1))
    if known_tools is not None:
        for c in candidates:
            if c in known_tools:
                return c
        return None
    if candidates:
        return candidates[0]
    return None


def extract_tool_name_strict(content: str, known_tools: Optional[Set[str]] = None) -> Optional[str]:
    if not isinstance(content, str):
        return None
    content = content.strip()
    if not content:
        return None
    decoder = json.JSONDecoder()
    for i, ch in enumerate(content):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(content[i:])
        except Exception:
            continue
        if isinstance(obj, dict):
            name = obj.get("name")
            if isinstance(name, str):
                if known_tools is None or name in known_tools:
                    return name
    return None


def extract_tool_name(content: str, known_tools: Optional[Set[str]] = None) -> Optional[str]:
    strict_name = extract_tool_name_strict(content, known_tools=known_tools)
    if strict_name is not None:
        return strict_name
    return extract_tool_name_relaxed(content, known_tools=known_tools)


def load_eval_dataset(dataset_path: str, limit: Optional[int] = None) -> List[Dict]:
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    samples = []
    for idx, item in enumerate(data):
        user_text = ""
        target_tool = None
        for msg in item.get("messages", []):
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "user":
                user_text = content
            elif role == "assistant":
                target_tool = extract_tool_name(content)
        if user_text and target_tool:
            samples.append({"idx": idx, "user": user_text, "label": target_tool})
        if limit is not None and len(samples) >= int(limit):
            break
    return samples


def parse_module_names(schema_path: str) -> List[str]:
    text = Path(schema_path).read_text(encoding="utf-8")
    return re.findall(r'<module\s+name="([a-zA-Z0-9_]+)">', text)


def load_vectorsearch_module(vectorsearch_path: str):
    spec = importlib.util.spec_from_file_location("vectorsearch_predictor", vectorsearch_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 predictor 模块: {vectorsearch_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_prompt_text(schema_name: str, tags: List[str], user_query: str) -> str:
    tag_lines = "\n".join([f"        <{x}/>" for x in tags])
    return f"""
        <prompt schema='{schema_name}'>
{tag_lines}
        <user>
            {user_query}
        </user>
        </prompt>
        """


@dataclass
class InferenceOutput:
    response: str
    predicted_tool: Optional[str]
    predicted_tool_relaxed: Optional[str]
    cache_time_ms: float
    active_kv_bytes: int
    baseline_allocated_bytes: Optional[int]
    absolute_peak_vram_bytes: Optional[int]
    peak_vram_bytes: Optional[int]
    prefill_latency_ms: float
    model_pipeline_ms: float
    model_time_to_first_token_ms: Optional[float]
    prompt_tokens: int


class PromptCacheRunner:
    def __init__(
        self,
        model_path: str,
        schema_path: str,
        max_ctx_length: int,
        schema_max_tokens: int,
        max_new_tokens: int,
        device_map: str,
        known_tools: Set[str],
    ):
        self.device_map = str(device_map)
        self.use_cuda = self.device_map.startswith("cuda")
        self.lm_for_cache = Qwen(
            str(model_path),
            device_map=self.device_map,
            **build_quantization_kwargs(True),
        )
        self.lm = self.lm_for_cache
        self.cache_engine = CacheEngine(int(max_ctx_length), self.lm_for_cache, target_device=None)
        self.gen_engine = GenerationEngine(self.lm)
        self.preproc = [CompactSpaces(), self.lm.get_formatter()]
        self.known_tools = set(known_tools)
        self.cache_engine.add_schema(read_file(str(schema_path), self.preproc), max_tokens=int(schema_max_tokens))
        self.parameter = GenerationParameters(
            temperature=0.0,
            repetition_penalty=1.0,
            top_p=1.0,
            top_k=-1,
            max_new_tokens=int(max_new_tokens),
            stop_token_ids=self.lm.stop_token_ids,
            stop_str=self.lm.stop_str + ["(END_JSON)", "<END_JSON>"],
        )

    def infer(self, prompt_text: str, disable_cache: bool) -> InferenceOutput:
        base_allocated_bytes = None
        if self.use_cuda:
            torch.cuda.synchronize()
            base_allocated_bytes = int(torch.cuda.memory_allocated())
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.nvtx.range_push("promptcache_model_pipeline")
        prompt = Prompt(prompt_text, self.preproc)
        t0 = time.perf_counter()
        token_ids, position_ids, cache_time, cache = self.cache_engine.process(
            prompt,
            no_cache=bool(disable_cache),
            return_full_position_ids=self.lm.use_full_position_ids,
        )
        active_kv_bytes = 0
        if cache is not None and len(cache) > 0:
            active_kv_bytes = int(
                sum(k.numel() * k.element_size() + v.numel() * v.element_size() for k, v in cache)
            )
        output_stream = self.gen_engine.generate(
            token_ids,
            position_ids,
            self.parameter,
            cache,
            stream_interval=2,
            use_full_position_ids=self.lm.use_full_position_ids,
        )
        resp = ""
        pre = 0
        first = True
        prefill_latency_ms = 0.0
        model_time_to_first_token_ms = None
        for outputs in output_stream:
            if first:
                first = False
                prefill_latency_ms = float(outputs.response_time)
                model_time_to_first_token_ms = (time.perf_counter() - t0) * 1000.0
            output_text = outputs.new_text
            now = len(output_text)
            if now > pre:
                resp += output_text[pre:now]
                pre = now
        resp = resp.replace("(END_JSON)", "").replace("<END_JSON>", "").rstrip()
        absolute_peak_vram_bytes = None
        peak_vram_bytes = None
        if self.use_cuda:
            torch.cuda.synchronize()
            absolute_peak_vram_bytes = int(torch.cuda.max_memory_allocated())
            peak_vram_bytes = max(absolute_peak_vram_bytes - base_allocated_bytes, 0)
            torch.cuda.nvtx.range_pop()
        model_pipeline_ms = (time.perf_counter() - t0) * 1000.0
        return InferenceOutput(
            response=resp,
            predicted_tool=extract_tool_name_strict(resp, known_tools=self.known_tools),
            predicted_tool_relaxed=extract_tool_name_relaxed(resp, known_tools=self.known_tools),
            cache_time_ms=float(cache_time),
            active_kv_bytes=active_kv_bytes,
            baseline_allocated_bytes=base_allocated_bytes,
            absolute_peak_vram_bytes=absolute_peak_vram_bytes,
            peak_vram_bytes=peak_vram_bytes,
            prefill_latency_ms=float(prefill_latency_ms),
            model_pipeline_ms=float(model_pipeline_ms),
            model_time_to_first_token_ms=model_time_to_first_token_ms,
            prompt_tokens=int(len(token_ids)),
        )


def aggregate_metrics(records: List[Dict]) -> Dict:
    total = len(records)
    ok = [x for x in records if "error" not in x]
    correct = sum(1 for x in ok if x.get("correct") is True)
    correct_relaxed = sum(1 for x in ok if x.get("correct_relaxed") is True)
    parsed = sum(1 for x in ok if x.get("predicted_tool") is not None)
    parsed_relaxed = sum(1 for x in ok if x.get("predicted_tool_relaxed") is not None)
    predictor_avail = [x for x in ok if x.get("predictor_top1") is not None]
    predictor_hit = sum(1 for x in predictor_avail if x.get("predictor_top1") == x.get("label"))
    predictor_hit_and_correct = sum(
        1 for x in predictor_avail if x.get("predictor_top1") == x.get("label") and x.get("correct") is True
    )
    e2e = [x["end_to_end_ms"] for x in ok if x.get("end_to_end_ms") is not None]
    prefill = [x["prefill_latency_ms"] for x in ok if x.get("prefill_latency_ms") is not None]
    model_pipeline = [x["model_pipeline_ms"] for x in ok if x.get("model_pipeline_ms") is not None]
    req_to_first_token = [x["request_to_first_token_ms"] for x in ok if x.get("request_to_first_token_ms") is not None]
    predictor_latency = [x["predictor_latency_ms"] for x in ok if x.get("predictor_latency_ms") is not None]
    cache_process = [x["cache_process_ms"] for x in ok if x.get("cache_process_ms") is not None]
    active_kv_mb = [x["active_kv_mb"] for x in ok if x.get("active_kv_mb") is not None]
    baseline_allocated_mb = [x["baseline_allocated_mb"] for x in ok if x.get("baseline_allocated_mb") is not None]
    absolute_peak_vram_mb = [x["absolute_peak_vram_mb"] for x in ok if x.get("absolute_peak_vram_mb") is not None]
    peak_vram_mb = [x["peak_vram_mb"] for x in ok if x.get("peak_vram_mb") is not None]
    out = {
        "samples_total": total,
        "samples_ok": len(ok),
        "samples_error": total - len(ok),
        "accuracy": (correct / len(ok)) if ok else 0.0,
        "accuracy_relaxed": (correct_relaxed / len(ok)) if ok else 0.0,
        "parse_rate": (parsed / len(ok)) if ok else 0.0,
        "parse_rate_relaxed": (parsed_relaxed / len(ok)) if ok else 0.0,
    }
    if predictor_avail:
        out["predictor_top1_accuracy"] = predictor_hit / len(predictor_avail)
        out["accuracy_when_predictor_correct"] = predictor_hit_and_correct / predictor_hit if predictor_hit else 0.0
    if e2e:
        out["end_to_end_ms_avg"] = float(np.mean(e2e))
        out["end_to_end_ms_p95"] = float(np.percentile(e2e, 95))
    if prefill:
        out["prefill_latency_ms_avg"] = float(np.mean(prefill))
        out["prefill_latency_ms_p95"] = float(np.percentile(prefill, 95))
    if model_pipeline:
        out["model_pipeline_ms_avg"] = float(np.mean(model_pipeline))
        out["model_pipeline_ms_p95"] = float(np.percentile(model_pipeline, 95))
    if req_to_first_token:
        out["request_to_first_token_ms_avg"] = float(np.mean(req_to_first_token))
        out["request_to_first_token_ms_p95"] = float(np.percentile(req_to_first_token, 95))
    if predictor_latency:
        out["predictor_latency_ms_avg"] = float(np.mean(predictor_latency))
        out["predictor_latency_ms_p95"] = float(np.percentile(predictor_latency, 95))
    if cache_process:
        out["cache_process_ms_avg"] = float(np.mean(cache_process))
        out["cache_process_ms_p95"] = float(np.percentile(cache_process, 95))
    if active_kv_mb:
        out["active_kv_mb_avg"] = float(np.mean(active_kv_mb))
        out["active_kv_mb_p95"] = float(np.percentile(active_kv_mb, 95))
    if baseline_allocated_mb:
        out["baseline_allocated_mb_avg"] = float(np.mean(baseline_allocated_mb))
        out["baseline_allocated_mb_p95"] = float(np.percentile(baseline_allocated_mb, 95))
    if absolute_peak_vram_mb:
        out["absolute_peak_vram_mb_avg"] = float(np.mean(absolute_peak_vram_mb))
        out["absolute_peak_vram_mb_p95"] = float(np.percentile(absolute_peak_vram_mb, 95))
        out["absolute_peak_vram_mb_max"] = float(np.max(absolute_peak_vram_mb))
    if peak_vram_mb:
        out["peak_vram_mb_avg"] = float(np.mean(peak_vram_mb))
        out["peak_vram_mb_p95"] = float(np.percentile(peak_vram_mb, 95))
        out["peak_vram_mb_max"] = float(np.max(peak_vram_mb))
    return out


def split_samples(samples: List[Dict], parts: int) -> List[List[Dict]]:
    shards = [[] for _ in range(parts)]
    for i, s in enumerate(samples):
        shards[i % parts].append(s)
    return shards


def warmup_predictor(predictor_module):
    for _ in range(10):
        predictor_module.model.encode(["warm up predictor"], convert_to_numpy=True, show_progress_bar=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def worker_run(worker_input: Dict) -> Dict:
    gpu_id = int(worker_input["gpu_id"])
    scenario_spec = worker_input["scenario_spec"]
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
    runner = PromptCacheRunner(
        model_path=worker_input["model_path"],
        schema_path=worker_input["schema_path"],
        max_ctx_length=int(worker_input["max_ctx_length"]),
        schema_max_tokens=int(worker_input["schema_max_tokens"]),
        max_new_tokens=int(worker_input["max_new_tokens"]),
        device_map=f"cuda:{gpu_id}",
        known_tools=set(worker_input["module_tags"]),
    )
    predictor_module = None
    predictor_tool_embeddings = None
    if worker_input["need_predictor"]:
        predictor_module = load_vectorsearch_module(worker_input["vectorsearch_path"])
        tool_texts = [f"{str(t['name']).replace('_', ' ')}: {t['description']}" for t in predictor_module.raw_tools]
        predictor_tool_embeddings = predictor_module.model.encode(tool_texts, convert_to_numpy=True, show_progress_bar=False)
        warmup_predictor(predictor_module)

    records = []
    samples = worker_input["samples"]
    all_tags = worker_input["module_tags"]
    warmup_ok = False
    warmup_error = None
    full_prompt_probe_tokens = None
    if samples:
        try:
            probe_prompt_text = build_prompt_text("function-call", list(all_tags), samples[0]["user"])
            probe_prompt = Prompt(probe_prompt_text, runner.preproc)
            probe_token_ids, _, _, _ = runner.cache_engine.process(
                probe_prompt,
                no_cache=True,
                return_full_position_ids=runner.lm.use_full_position_ids,
            )
            full_prompt_probe_tokens = int(len(probe_token_ids))
            max_ctx_length = int(worker_input["max_ctx_length"])
            if full_prompt_probe_tokens >= max_ctx_length:
                print(
                    f"[worker cuda:{gpu_id}] WARNING full-prompt tokens={full_prompt_probe_tokens} >= max_ctx_length={max_ctx_length}, full_gpu_cache may fail",
                    flush=True,
                )
        except Exception:
            full_prompt_probe_tokens = None
        try:
            warmup_tags = list(all_tags)
            if scenario_spec["use_predictor"] and predictor_module is not None and predictor_tool_embeddings is not None:
                warmup_q = predictor_module.preprocess_text(samples[0]["user"])
                warmup_q_emb = predictor_module.model.encode([warmup_q], convert_to_numpy=True, show_progress_bar=False)
                warmup_sims = cosine_similarity(warmup_q_emb, predictor_tool_embeddings)[0]
                warmup_top_idx = int(np.argmax(warmup_sims))
                warmup_tags = [predictor_module.raw_tools[warmup_top_idx]["name"]]
            warmup_prompt = build_prompt_text("function-call", warmup_tags, samples[0]["user"])
            _ = runner.infer(prompt_text=warmup_prompt, disable_cache=False)
            if scenario_spec["use_predictor"] and (not scenario_spec["disable_cache"]):
                for tag in all_tags:
                    warmup_prompt = build_prompt_text("function-call", [tag], samples[0]["user"])
                    _ = runner.infer(prompt_text=warmup_prompt, disable_cache=False)
            warmup_ok = True
        except Exception as e:
            warmup_error = f"{type(e).__name__}: {e}"
    for i, sample in enumerate(samples):
        user_text = sample["user"]
        label = str(sample["label"]).strip()
        pred_top1 = None
        try:
            predictor_latency_ms = 0.0
            req_start = time.perf_counter()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.nvtx.range_push("e2e_user_to_output")
            if scenario_spec["use_predictor"]:
                pred_start = time.perf_counter()
                q = predictor_module.preprocess_text(user_text)
                q_emb = predictor_module.model.encode([q], convert_to_numpy=True, show_progress_bar=False)
                sims = cosine_similarity(q_emb, predictor_tool_embeddings)[0]
                top_idx = int(np.argmax(sims))
                pred_top1 = predictor_module.raw_tools[top_idx]["name"]
                predictor_latency_ms = (time.perf_counter() - pred_start) * 1000.0
            tags = [pred_top1] if scenario_spec["use_predictor"] else list(all_tags)
            prompt_text = build_prompt_text("function-call", tags, user_text)
            out = runner.infer(prompt_text=prompt_text, disable_cache=scenario_spec["disable_cache"])
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.nvtx.range_pop()
            req_e2e_ms = (time.perf_counter() - req_start) * 1000.0
            request_to_first_token_ms = None
            cache_process_ms = float(out.cache_time_ms)
            if out.model_time_to_first_token_ms is not None:
                request_to_first_token_ms = predictor_latency_ms + float(out.model_time_to_first_token_ms)
            rec = {
                "scenario": scenario_spec["name"],
                "group": scenario_spec["group"],
                "idx": sample["idx"],
                "worker_gpu": gpu_id,
                "worker_sample_no": i,
                "user": user_text,
                "label": label,
                "predictor_top1": pred_top1,
                "prompt_tags": tags,
                "response": out.response,
                "predicted_tool": out.predicted_tool,
                "predicted_tool_relaxed": out.predicted_tool_relaxed,
                "correct": out.predicted_tool == label,
                "correct_relaxed": out.predicted_tool_relaxed == label,
                "predictor_latency_ms": predictor_latency_ms,
                "cache_time_ms": out.cache_time_ms,
                "cache_process_ms": cache_process_ms,
                "active_kv_bytes": out.active_kv_bytes,
                "active_kv_mb": out.active_kv_bytes / (1024.0 * 1024.0),
                "baseline_allocated_bytes": out.baseline_allocated_bytes,
                "baseline_allocated_mb": (
                    out.baseline_allocated_bytes / (1024.0 * 1024.0)
                    if out.baseline_allocated_bytes is not None else None
                ),
                "absolute_peak_vram_bytes": out.absolute_peak_vram_bytes,
                "absolute_peak_vram_mb": (
                    out.absolute_peak_vram_bytes / (1024.0 * 1024.0)
                    if out.absolute_peak_vram_bytes is not None else None
                ),
                "peak_vram_bytes": out.peak_vram_bytes,
                "peak_vram_mb": (
                    out.peak_vram_bytes / (1024.0 * 1024.0) if out.peak_vram_bytes is not None else None
                ),
                "prefill_latency_ms": out.prefill_latency_ms,
                "request_to_first_token_ms": request_to_first_token_ms,
                "model_pipeline_ms": out.model_pipeline_ms,
                "end_to_end_ms": req_e2e_ms,
                "prompt_tokens": out.prompt_tokens,
            }
        except Exception as e:
            if torch.cuda.is_available():
                try:
                    torch.cuda.synchronize()
                    torch.cuda.nvtx.range_pop()
                except Exception:
                    pass
            rec = {
                "scenario": scenario_spec["name"],
                "group": scenario_spec["group"],
                "idx": sample["idx"],
                "worker_gpu": gpu_id,
                "worker_sample_no": i,
                "user": user_text,
                "label": label,
                "predictor_top1": pred_top1,
                "error": f"{type(e).__name__}: {e}",
            }
        records.append(rec)
        if (i + 1) % 10 == 0:
            print(f"[worker cuda:{gpu_id}][{scenario_spec['name']}] processed={i + 1}/{len(samples)}", flush=True)
    return {
        "gpu_id": gpu_id,
        "scenario_name": scenario_spec["name"],
        "records": records,
        "full_prompt_probe_tokens": full_prompt_probe_tokens,
        "full_warmup_ok": warmup_ok,
        "full_warmup_error": warmup_error,
    }


def write_json(path: Path, obj: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows: List[Dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_markdown_report(summary: Dict) -> str:
    lines = []
    lines.append("# Function Call 实验报告")
    lines.append("")
    lines.append("## 指标表")
    lines.append("")
    lines.append("| 分组 | 场景 | 端到端延迟(ms, avg) | 请求到首token(ms, avg) | Predictor(ms, avg) | Cache Process(ms, avg) | 推理准确率(Strict) | 推理准确率(Relaxed) | Parse率(Strict) | Parse率(Relaxed) | 活跃KV(MB, avg) | 基线显存(MB, avg) | 绝对峰值显存(MB, avg) | 绝对峰值显存(MB, max) | 请求增量峰值显存(MB, avg) | 请求增量峰值显存(MB, max) | Prefill Latency(ms, avg) | 有效样本/总样本 |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for item in summary["scenarios"]:
        m = item["metrics"]
        lines.append(
            f"| {item['group']} | {item['name']} | {m.get('end_to_end_ms_avg', float('nan')):.4f} | {m.get('request_to_first_token_ms_avg', float('nan')):.4f} | {m.get('predictor_latency_ms_avg', 0.0):.4f} | {m.get('cache_process_ms_avg', float('nan')):.4f} | {m.get('accuracy', 0.0):.4%} | {m.get('accuracy_relaxed', 0.0):.4%} | {m.get('parse_rate', 0.0):.4%} | {m.get('parse_rate_relaxed', 0.0):.4%} | {m.get('active_kv_mb_avg', float('nan')):.4f} | {m.get('baseline_allocated_mb_avg', float('nan')):.4f} | {m.get('absolute_peak_vram_mb_avg', float('nan')):.4f} | {m.get('absolute_peak_vram_mb_max', float('nan')):.4f} | {m.get('peak_vram_mb_avg', float('nan')):.4f} | {m.get('peak_vram_mb_max', float('nan')):.4f} | {m.get('prefill_latency_ms_avg', float('nan')):.4f} | {m.get('samples_ok', 0)}/{m.get('samples_total', 0)} |"
        )
    lines.append("")
    lines.append("## 说明")
    lines.append("")
    lines.append("- 实验目标：predictor_gpu_cache")
    lines.append("- 对比基线：full_gpu_no_cache")
    lines.append("- 消融：full_gpu_cache、predictor_gpu_no_cache")
    lines.append("- 每个实验固定一张GPU，实验开始前执行一次full prompt warmup（不计入结果）")
    lines.append("- 端到端延迟在 NVTX 区间 `e2e_user_to_output` 内统计，口径为 user请求进入(含predictor)到模型输出结束，不含add_schema预计算")
    return "\n".join(lines) + "\n"


def parse_gpu_ids(gpu_ids) -> List[int]:
    if isinstance(gpu_ids, int):
        return [gpu_ids]
    if isinstance(gpu_ids, (list, tuple)):
        out = []
        for x in gpu_ids:
            if isinstance(x, int):
                out.append(x)
            else:
                s = str(x).strip()
                if s:
                    out.append(int(s))
        return out
    s = str(gpu_ids).strip()
    s = s.replace("[", "").replace("]", "").replace("(", "").replace(")", "")
    parts = [x.strip() for x in s.split(",") if x.strip()]
    return [int(x) for x in parts]


def main(
    dataset_path="/data/xingye/data/glaive-function-calling-v2/model_training/tool_predict/tool_selection_train.json",
    schema_path="/home/xingye/prompt-cache/function_call/schema/function.xml",
    vectorsearch_path="/home/xingye/prompt-cache/function_call/predictor/vectorsearch.py",
    output_dir="/home/xingye/prompt-cache/function_call/test/outputs",
    model_path="/data/zy/models/Qwen/Qwen3/Qwen3-8B-Base",
    limit=None,
    max_ctx_length=8192,
    schema_max_tokens=8192,
    max_new_tokens=256,
    gpu_ids="0,1,2,3,4",
):
    samples = load_eval_dataset(dataset_path, limit=limit)
    if not samples:
        raise RuntimeError("没有加载到可用样本")
    module_tags = parse_module_names(schema_path)
    if not module_tags:
        raise RuntimeError(f"未从 schema 解析到 module: {schema_path}")
    gpu_list = parse_gpu_ids(gpu_ids)
    if not gpu_list:
        raise RuntimeError("gpu_ids 不能为空")
    workers = min(len(gpu_list), len(SCENARIOS))
    active_gpus = gpu_list[:workers]
    jobs = []
    scenario_gpu_binding = {}
    for i, spec in enumerate(SCENARIOS):
        gpu = active_gpus[i % len(active_gpus)]
        scenario_gpu_binding[spec["name"]] = gpu
        jobs.append(
            {
                "gpu_id": gpu,
                "scenario_spec": spec,
                "samples": samples,
                "model_path": model_path,
                "schema_path": schema_path,
                "vectorsearch_path": vectorsearch_path,
                "module_tags": module_tags,
                "max_ctx_length": int(max_ctx_length),
                "schema_max_tokens": int(schema_max_tokens),
                "max_new_tokens": int(max_new_tokens),
                "need_predictor": bool(spec["use_predictor"]),
            }
        )

    merged = {x["name"]: [] for x in SCENARIOS}
    probe_tokens = []
    full_warmup_status = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(worker_run, j) for j in jobs]
        for fu in concurrent.futures.as_completed(futures):
            out = fu.result()
            scenario_name = out["scenario_name"]
            merged[scenario_name] = out["records"]
            if out.get("full_prompt_probe_tokens") is not None:
                probe_tokens.append(int(out["full_prompt_probe_tokens"]))
            full_warmup_status[scenario_name] = {
                "ok": bool(out.get("full_warmup_ok", False)),
                "error": out.get("full_warmup_error"),
            }

    for name in merged:
        merged[name] = sorted(merged[name], key=lambda x: x.get("idx", 10**12))

    scenarios_summary = []
    for spec in SCENARIOS:
        name = spec["name"]
        scenarios_summary.append(
            {
                "name": name,
                "group": spec["group"],
                "metrics": aggregate_metrics(merged[name]),
            }
        )

    out_dir = Path(output_dir)
    summary = {
        "config": {
            "dataset_path": dataset_path,
            "schema_path": schema_path,
            "vectorsearch_path": vectorsearch_path,
            "model_path": model_path,
            "samples": len(samples),
            "gpu_ids": active_gpus,
            "scenario_gpu_binding": scenario_gpu_binding,
            "full_prompt_probe_tokens_max": max(probe_tokens) if probe_tokens else None,
            "full_warmup_status": full_warmup_status,
            "note": "每个实验固定一张GPU；每个实验开始前先做一次full prompt warmup（不计入统计）；每个worker内add_schema仅执行一次；实验目标为predictor_gpu_cache。",
        },
        "scenarios": scenarios_summary,
    }
    write_json(out_dir / "summary.json", summary)
    for spec in SCENARIOS:
        write_jsonl(out_dir / f"{spec['name']}.jsonl", merged[spec["name"]])
    (out_dir / "report.md").write_text(build_markdown_report(summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"Markdown report saved to: {out_dir / 'report.md'}", flush=True)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
