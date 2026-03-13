"""
generate_question_embed.py
──────────────────────────
Re-generate GrailQA question embeddings using text-embedding-3-small (1536d).
Reads grailqa_dev_pyql.json → outputs grailqa_question_embed_1536.json.

Output format (same as original grailqa_question_embed.json):
    { "question string": [0.12, 0.34, ...], ... }

Usage:
    python generate_question_embed.py \
        --input_json  ../../data/GrailQA_v1.0/grailqa_dev_pyql.json \
        --output_json ../../data/grailqa_question_embed_1536.json \
        --api_key     sk-xxx
        --api_base    https://api.openai.com/v1

Cost: ~6,763 dev questions x ~15 tokens ≈ 100K tokens → ~$0.002
"""

import json, argparse, time, os
from openai import OpenAI
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--input_json",  default="../../data/GrailQA_v1.0/grailqa_dev_pyql.json")
parser.add_argument("--output_json", default="../../data/grailqa_question_embed_1536.json")
parser.add_argument("--api_key",     default=None)
parser.add_argument("--api_base",    default="https://api.openai.com/v1")
parser.add_argument("--model",       default="text-embedding-3-small")
parser.add_argument("--batch_size",  type=int, default=100)
parser.add_argument("--resume",      action="store_true")
args = parser.parse_args()

client = OpenAI(
    api_key=args.api_key or os.environ.get("OPENAI_API_KEY"),
    base_url=args.api_base
)

# ── load questions from grailqa_dev_pyql.json ─────────────────────────────────
print(f"[1/3] Loading questions from {args.input_json} ...")
with open(args.input_json, "r") as f:
    data = json.load(f)

# grailqa_dev_pyql.json is a list of dicts with key "question"
questions = list({item["question"] for item in data if "question" in item})
print(f"      {len(questions)} unique questions found.")

# ── resume support ────────────────────────────────────────────────────────────
result = {}
if args.resume and os.path.exists(args.output_json):
    with open(args.output_json, "r") as f:
        result = json.load(f)
    print(f"      Resume mode: {len(result)} already done, skipping.")

todo = [q for q in questions if q not in result]
print(f"      To embed: {len(todo)} questions.")

# ── embed ─────────────────────────────────────────────────────────────────────
print(f"[2/3] Embedding with '{args.model}' (batch_size={args.batch_size}) ...")

def embed_batch(texts):
    while True:
        try:
            resp = client.embeddings.create(model=args.model, input=texts)
            return [x.embedding for x in sorted(resp.data, key=lambda x: x.index)]
        except Exception as e:
            if "rate" in str(e).lower():
                print(f"\n  [Rate limit] sleeping 10s ...")
                time.sleep(10)
            else:
                print(f"\n  [Error] {e}")
                raise

for i in tqdm(range(0, len(todo), args.batch_size)):
    batch = todo[i : i + args.batch_size]
    embeddings = embed_batch(batch)
    for q, emb in zip(batch, embeddings):
        result[q] = emb
    # checkpoint every 1000
    if (i // args.batch_size) % 10 == 9:
        with open(args.output_json, "w") as f:
            json.dump(result, f)
        print(f"  [checkpoint] {len(result)}/{len(questions)} saved.")

# ── save final ────────────────────────────────────────────────────────────────
print(f"[3/3] Saving to {args.output_json} ...")
with open(args.output_json, "w") as f:
    json.dump(result, f)

dims = len(next(iter(result.values())))
print(f"\n✅ Done. {len(result)} questions embedded, dim={dims}.")
print(f"   Output: {args.output_json}")