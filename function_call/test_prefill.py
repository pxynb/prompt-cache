import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import fire

sys.path.insert(1, str(Path(__file__).resolve().parents[1]))

import demo


def _load_json_array(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_latency_map(items: List[Dict[str, Any]]) -> Dict[int, float]:
    m: Dict[int, float] = {}
    for it in items:
        idx = it.get("idx")
        lat = it.get("prefill_latency_ms")
        if isinstance(idx, int) and isinstance(lat, (int, float)) and lat > 0 and "error" not in it:
            m[idx] = float(lat)
    return m


def _ensure_out_dir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def main(
    limit: Optional[int] = None,
    start: int = 0,
    out_dir: str = "/home/xingye/prompt-cache/function_call/outputs",
    max_new_tokens: int = 256,
    max_ctx_length: int = 4096,
    schema_max_tokens: int = 2048,
    model_name: str = "/data/zy/models/Qwen/Qwen3/Qwen3-8B-Base",
    print_every: int = 1,
) -> Tuple[str, str, str]:
    out_dir_p = _ensure_out_dir(out_dir)

    suffix = f"start{start}_limit{limit if limit is not None else 'all'}"
    cache_out = str(out_dir_p / f"qwen_enable_cache_true_outputs_{suffix}.json")
    nocache_out = str(out_dir_p / f"qwen_enable_cache_false_outputs_{suffix}.json")
    speedup_out = str(out_dir_p / f"qwen_prefill_speedup_{suffix}.json")

    demo.main(
        enable_cache=True,
        max_ctx_length=max_ctx_length,
        schema_max_tokens=schema_max_tokens,
        max_new_tokens=max_new_tokens,
        model_name=model_name,
        output_path=cache_out,
        start=start,
        limit=limit,
        print_every=print_every,
    )

    demo.main(
        enable_cache=False,
        max_ctx_length=max_ctx_length,
        schema_max_tokens=schema_max_tokens,
        max_new_tokens=max_new_tokens,
        model_name=model_name,
        output_path=nocache_out,
        start=start,
        limit=limit,
        print_every=print_every,
    )

    cache_items = _load_json_array(cache_out)
    nocache_items = _load_json_array(nocache_out)

    cache_map = _build_latency_map(cache_items)
    nocache_map = _build_latency_map(nocache_items)

    all_idxs = sorted(set(cache_map.keys()) & set(nocache_map.keys()))
    per_item: List[Dict[str, Any]] = []
    speedups: List[float] = []
    for idx in all_idxs:
        cache_lat = cache_map[idx]
        nocache_lat = nocache_map[idx]
        speedup = nocache_lat / cache_lat if cache_lat > 0 else None
        if isinstance(speedup, (int, float)) and speedup > 0:
            speedups.append(float(speedup))
        per_item.append(
            {
                "idx": idx,
                "prefill_latency_cache_ms": cache_lat,
                "prefill_latency_no_cache_ms": nocache_lat,
                "speedup": speedup,
            }
        )

    summary = {
        "count_compared": len(all_idxs),
        "count_cache_ok": len(cache_map),
        "count_no_cache_ok": len(nocache_map),
        "mean_speedup": (sum(speedups) / len(speedups)) if speedups else None,
        "min_speedup": min(speedups) if speedups else None,
        "max_speedup": max(speedups) if speedups else None,
        "cache_outputs": cache_out,
        "no_cache_outputs": nocache_out,
    }

    payload = {"summary": summary, "items": per_item}
    with open(speedup_out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"Saved cache outputs to: {cache_out}", flush=True)
    print(f"Saved no-cache outputs to: {nocache_out}", flush=True)
    print(f"Saved speedup report to: {speedup_out}", flush=True)
    print(f"Compared {summary['count_compared']} items; mean speedup={summary['mean_speedup']}", flush=True)

    return cache_out, nocache_out, speedup_out


if __name__ == "__main__":
    fire.Fire(main)
