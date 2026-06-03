import argparse
import json
from pathlib import Path
from statistics import mean


def read_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def percentile(values, p):
    if not values:
        return None
    if p <= 0:
        return min(values)
    if p >= 100:
        return max(values)
    arr = sorted(values)
    pos = (len(arr) - 1) * (p / 100.0)
    lo = int(pos)
    hi = min(lo + 1, len(arr) - 1)
    frac = pos - lo
    return arr[lo] * (1.0 - frac) + arr[hi] * frac


def values_of(rows, key):
    vals = []
    for r in rows:
        v = r.get(key)
        if isinstance(v, (int, float)):
            vals.append(float(v))
    return vals


def typical_mean_upper_cut(values, keep_percent=95.0):
    if not values:
        return None, None
    cut = percentile(values, keep_percent)
    kept = [v for v in values if v <= cut]
    if not kept:
        return None, cut
    return mean(kept), cut


def fmt_num(v, digits=4):
    if v is None:
        return "-"
    return f"{v:.{digits}f}"


def fmt_pct(v):
    if v is None:
        return "-"
    return f"{v * 100:.4f}%"


def scenario_metrics(rows):
    total = len(rows)
    ok = sum(1 for r in rows if r.get("predicted_tool") is not None)
    acc = mean([1.0 if r.get("correct") else 0.0 for r in rows]) if rows else None
    parse_rate = mean([1.0 if r.get("predicted_tool") else 0.0 for r in rows]) if rows else None
    lat_keys = [
        "end_to_end_ms",
        "request_to_first_token_ms",
        "prefill_latency_ms",
        "predictor_latency_ms",
        "cache_process_ms",
        "peak_vram_mb",
    ]
    out = {
        "samples_total": total,
        "samples_ok": ok,
        "accuracy": acc,
        "parse_rate": parse_rate,
    }
    for k in lat_keys:
        vals = values_of(rows, k)
        typ_avg, _ = typical_mean_upper_cut(vals, keep_percent=95.0)
        out[f"{k}_typical_avg"] = typ_avg
        out[f"{k}_p50"] = percentile(vals, 50.0)
        out[f"{k}_p90"] = percentile(vals, 90.0)
    return out


def build_markdown(summary, scenarios_rows):
    lines = []
    lines.append("# Function Call 常规情况报告（去除长尾影响）")
    lines.append("")
    lines.append("## 指标口径")
    lines.append("")
    lines.append("- 常规均值(typical avg)：对每个场景按指标去掉最高 5% 样本后计算均值")
    lines.append("- p50 / p90：展示中位数与常规高位，不展示 p95/p99/max")
    lines.append("- 准确率与 Parse 率保持全量样本统计")
    lines.append("")
    lines.append("## 常规指标表")
    lines.append("")
    lines.append("| 分组 | 场景 | E2E(ms, typical avg) | E2E p50 | E2E p90 | 首token(ms, typical avg) | 首token p50 | 首token p90 | Prefill(ms, typical avg) | Predictor(ms, typical avg) | Cache(ms, typical avg) | 增量峰值显存(MB, typical avg) | 准确率(Strict) | Parse率(Strict) | 有效样本/总样本 |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    ordered = summary.get("scenarios", [])
    name_to_group = {x.get("name"): x.get("group", "-") for x in ordered}
    scenario_names = [x.get("name") for x in ordered if x.get("name")]
    for n in scenarios_rows:
        if n not in scenario_names:
            scenario_names.append(n)
    for name in scenario_names:
        rows = scenarios_rows.get(name, [])
        m = scenario_metrics(rows)
        group = name_to_group.get(name, rows[0].get("group", "-") if rows else "-")
        lines.append(
            f"| {group} | {name} | {fmt_num(m.get('end_to_end_ms_typical_avg'))} | {fmt_num(m.get('end_to_end_ms_p50'))} | {fmt_num(m.get('end_to_end_ms_p90'))} | "
            f"{fmt_num(m.get('request_to_first_token_ms_typical_avg'))} | {fmt_num(m.get('request_to_first_token_ms_p50'))} | {fmt_num(m.get('request_to_first_token_ms_p90'))} | "
            f"{fmt_num(m.get('prefill_latency_ms_typical_avg'))} | {fmt_num(m.get('predictor_latency_ms_typical_avg'))} | {fmt_num(m.get('cache_process_ms_typical_avg'))} | "
            f"{fmt_num(m.get('peak_vram_mb_typical_avg'))} | {fmt_pct(m.get('accuracy'))} | {fmt_pct(m.get('parse_rate'))} | {m.get('samples_ok', 0)}/{m.get('samples_total', 0)} |"
        )
    lines.append("")
    lines.append("## 说明")
    lines.append("")
    lines.append("- 本报告用于观察常规请求表现，不用于尾延迟分析")
    lines.append("- 若要评估最坏情况，请结合原始 report.md 中的 p95 / max 指标")
    return "\n".join(lines) + "\n"


def load_rows_by_scenario(outputs_dir: Path):
    rows = {}
    for p in sorted(outputs_dir.glob("*.jsonl")):
        rows[p.stem] = read_jsonl(p)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outputs-dir",
        default="/home/xingye/prompt-cache/function_call/test/outputs_all",
    )
    parser.add_argument(
        "--out-report",
        default=None,
    )
    args = parser.parse_args()
    outputs_dir = Path(args.outputs_dir)
    summary_path = outputs_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"未找到 summary.json: {summary_path}")
    summary = read_json(summary_path)
    rows_by_scenario = load_rows_by_scenario(outputs_dir)
    report = build_markdown(summary, rows_by_scenario)
    out_report = Path(args.out_report) if args.out_report else outputs_dir / "report_typical.md"
    out_report.write_text(report, encoding="utf-8")
    print(f"已生成常规报告: {out_report}")


if __name__ == "__main__":
    main()
