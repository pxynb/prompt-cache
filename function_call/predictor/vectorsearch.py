import json
import random
import re
import time
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# --- 环境检查与模型加载 ---
device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"正在准备环境... 检测到设备: {device_str.upper()}")

model = SentenceTransformer('all-MiniLM-L6-v2', device=device_str)

raw_tools = [
    {"name": "generate_password", "description": "Generate a random password"},
    {"name": "calculate_loan_payment", "description": "Calculate the monthly payment for a loan"},
    {"name": "calculate_tip", "description": "Calculate the tip amount for a bill"},
    {"name": "get_stock_price", "description": "Get the current stock price"},
    {"name": "generate_random_number", "description": "Generate a random number within a given range"},
    {"name": "search_books", "description": "Search for books based on title, author, or genre"},
    {"name": "convert_currency", "description": "Convert currency from one unit to another"},
    {"name": "calculate_distance", "description": "Calculate the distance between two points"},
    {"name": "get_movie_details", "description": "Get details of a movie"},
    {"name": "search_movies", "description": "Search for movies based on title or genre"},
    {"name": "calculate_age", "description": "Calculate the age based on date of birth"},
    {"name": "calculate_discount", "description": "Calculate the discounted price of a product"},
    {"name": "calculate_bmi", "description": "Calculate the Body Mass Index (BMI)"},
    {"name": "calculate_area", "description": "Calculate the area of a shape"},
    {"name": "search_recipes", "description": "Search for recipes based on ingredients"},
    {"name": "generate_qr_code", "description": "Generate a QR code for a given text"},
    {"name": "generate_random_password", "description": "Generate a random password with specified criteria"},
    {"name": "create_todo", "description": "Create a new todo item"},
    {"name": "create_calendar_event", "description": "Create a new calendar event"},
    {"name": "send_email", "description": "Send an email to a specified recipient"}
]

def preprocess_text(text):
    if not isinstance(text, str): return ""
    return text.lower().strip()

def extract_function_name(content):
    content = content.strip()
    if "<tool_call>" in content:
        try:
            json_str = content.split("<tool_call>")[1].split("</tool_call>")[0].strip()
            return json.loads(json_str).get("name")
        except: pass
    match = re.search(r'\{.*\}', content, re.DOTALL)
    if match:
        try:
            return json.loads(match.group()).get("name")
        except: pass
    return None

def load_dataset(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"错误: 找不到文件 {file_path}")
        return []
    
    samples = []
    for item in data:
        messages = item.get("messages", [])
        user_query, target_tool = "", None
        for msg in messages:
            if msg["role"] == "user": user_query = msg["content"]
            elif msg["role"] == "assistant": target_tool = extract_function_name(msg["content"])
        if user_query and target_tool:
            samples.append({"query": user_query, "label": target_tool})
    return samples

def run_embedding_experiment(data_path):
    all_samples = load_dataset(data_path)
    if not all_samples: return

    random.seed(42)
    random.shuffle(all_samples)
    split_idx = int(len(all_samples) * 0.8)
    val_samples = all_samples[split_idx:]

    current_device = model.device
    is_cuda = current_device.type == 'cuda'

    print(f"正在编码工具库向量 (运行设备: {current_device})...")
    tool_texts = [f"{str(t['name']).replace('_', ' ')}: {t['description']}" for t in raw_tools]
    tool_embeddings = model.encode(tool_texts, convert_to_numpy=True)

    # --- (Warm-up) ---
    print(f"正在进行 {current_device.type.upper()} 预热...")
    for _ in range(15):
        _ = model.encode(["warm up text"], convert_to_numpy=True, show_progress_bar=False)
    if is_cuda: torch.cuda.synchronize() 

    correct_t1, correct_t3 = 0, 0
    latencies = []

    print(f"开始测试，验证集规模: {len(val_samples)} 条样本...")
    
    for sample in val_samples:
        query_text = preprocess_text(sample["query"])
        
        # 计时开始
        if is_cuda: torch.cuda.synchronize()
        start_time = time.perf_counter()
        
        # 推理核心
        query_embedding = model.encode([query_text], convert_to_numpy=True, show_progress_bar=False)
        similarities = cosine_similarity(query_embedding, tool_embeddings)[0]
        top_indices = np.argsort(similarities)[::-1]
        
        # 计时结束
        if is_cuda: torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        latencies.append(end_time - start_time)

        # 统计准确率
        pred_t1 = raw_tools[top_indices[0]]["name"]
        if pred_t1 == sample["label"]: correct_t1 += 1
        
        pred_t3 = [raw_tools[idx]["name"] for idx in top_indices[:3]]
        if sample["label"] in pred_t3: correct_t3 += 1
        
    # --- 4. 统计结果 ---
    avg_latency = (sum(latencies) / len(latencies)) * 1000 
    p95_latency = np.percentile(latencies, 95) * 1000
    p99_latency = np.percentile(latencies, 99) * 1000
    tps = 1.0 / (sum(latencies) / len(latencies))

    print(f"\n" + "="*40)
    print(f"  推理实验结果")
    print(f"  运行设备: {current_device}")
    print(f"  模型: all-MiniLM-L6-v2")
    print(f"  样本数: {len(val_samples)}")
    print(f"  Top-1 准确率: {correct_t1 / len(val_samples):.2%}")
    print(f"  Top-3 准确率: {correct_t3 / len(val_samples):.2%}")
    print("-" * 40)
    print(f"  平均耗时: {avg_latency:.4f} ms")
    print(f"  P95 耗时: {p95_latency:.4f} ms")
    print(f"  P99 耗时: {p99_latency:.4f} ms")
    print(f"  吞吐量 (TPS): {tps:.2f} req/sec")
    print("="*40)

if __name__ == "__main__":
    DATA_PATH = '/data/xingye/data/glaive-function-calling-v2/dataset_qwen.json'
    run_embedding_experiment(DATA_PATH)