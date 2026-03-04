import json, os, time
from typing import Dict, List
from openai import OpenAI

DATA_DIR = "data"
REL_TXT = os.path.join(DATA_DIR, "fb_relation.txt")
REL_OUT = os.path.join(DATA_DIR, "fb_relation_embed.json")
GRAIL_IN = os.path.join(DATA_DIR, "GrailQA_v1.0", "grailqa_dev_pyql.json")
GRAIL_OUT = os.path.join(DATA_DIR, "grailqa_question_embed.json")

# ====== 这里填你的 embedding 配置（推荐用 OpenAI 官方 embedding）======
EMB_API_BASE = "https://api.openai.com/v1"
EMB_API_KEY  = os.environ.get("OPENAI_API_KEY", "")  # 建议用环境变量
EMB_MODEL    = "text-embedding-3-small"
client = OpenAI(api_key=EMB_API_KEY, base_url=EMB_API_BASE)
# ================================================================

def load_json(path: str, default):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return default

def save_json(path: str, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
    os.replace(tmp, path)

def batch(iterable, n=128):
    buf = []
    for x in iterable:
        buf.append(x)
        if len(buf) >= n:
            yield buf
            buf = []
    if buf:
        yield buf

def get_questions_from_grail(path: str) -> List[str]:
    data = load_json(path, [])
    qs = []
    # 常见字段名兼容
    for item in data:
        for k in ["question", "Question", "query", "utterance"]:
            if k in item and isinstance(item[k], str):
                qs.append(item[k])
                break
    # 去重但保序
    seen = set()
    out = []
    for q in qs:
        if q not in seen:
            seen.add(q)
            out.append(q)
    return out

def embed_texts(texts: List[str]) -> List[List[float]]:
    resp = client.embeddings.create(model=EMB_MODEL, input=texts)
    return [d.embedding for d in resp.data]

def main():
    if not EMB_API_KEY:
        raise SystemExit("请先设置 OPENAI_API_KEY 环境变量（或把 key 直接填在脚本里）")

    # 1) relations
    rel_cache: Dict[str, List[float]] = load_json(REL_OUT, {})
    with open(REL_TXT, "r", encoding="utf-8") as f:
        rels = [line.strip() for line in f if line.strip()]
    todo_rels = [r for r in rels if r not in rel_cache]
    print(f"[REL] total={len(rels)} cached={len(rel_cache)} todo={len(todo_rels)}")

    for chunk in batch(todo_rels, 128):
        embs = embed_texts(chunk)
        for r, e in zip(chunk, embs):
            rel_cache[r] = e
        save_json(REL_OUT, rel_cache)
        time.sleep(0.2)  # 温和一点，避免限速

    # 2) grail questions
    q_cache: Dict[str, List[float]] = load_json(GRAIL_OUT, {})
    questions = get_questions_from_grail(GRAIL_IN)
    todo_q = [q for q in questions if q not in q_cache]
    print(f"[GRAIL-Q] total={len(questions)} cached={len(q_cache)} todo={len(todo_q)}")

    for chunk in batch(todo_q, 128):
        embs = embed_texts(chunk)
        for q, e in zip(chunk, embs):
            q_cache[q] = e
        save_json(GRAIL_OUT, q_cache)
        time.sleep(0.2)

    print("✅ Done.")
    print(f"Saved: {REL_OUT}")
    print(f"Saved: {GRAIL_OUT}")

if __name__ == "__main__":
    main()
