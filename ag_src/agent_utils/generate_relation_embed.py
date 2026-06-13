"""
generate_relation_embed.py
──────────────────────────
Re-generate Freebase relation embeddings using text-embedding-3-small (1536d).
Reads fb_relation.txt → outputs fb_relation_embed_1536.json.

Usage:
    python generate_relation_embed.py \
        --relation_txt ../../data/fb_relation.txt \
        --output_json  ../../data/fb_relation_embed_1536.json \
        --api_key      sk-xxx
        --api_base     https://api.openai.com/v1   # or Michelle's endpoint

Cost: ~24,436 relations x ~8 tokens ≈ 200K tokens → ~$0.004
"""

import json, argparse, time, os
from openai import OpenAI
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--relation_txt", default="../../data/fb_relation.txt")
parser.add_argument("--output_json",  default="../../data/fb_relation_embed_1536.json")
parser.add_argument("--api_key",      default=None)
parser.add_argument("--api_base",     default="https://api.openai.com/v1")
parser.add_argument("--model",        default="text-embedding-3-small")
parser.add_argument("--batch_size",   type=int, default=100)
parser.add_argument("--resume",       action="store_true", help="skip already-embedded relations")
args = parser.parse_args()

client = OpenAI(
    api_key=args.api_key or os.environ.get("OPENAI_API_KEY"),
    base_url=args.api_base
)

# ── load relations ────────────────────────────────────────────────────────────
print(f"[1/3] Loading relations from {args.relation_txt} ...")
with open(args.relation_txt, "r") as f:
    relations = [line.strip() for line in f if line.strip()]
print(f"      {len(relations)} relations found.")

# ── resume support ────────────────────────────────────────────────────────────
result = {}
if args.resume and os.path.exists(args.output_json):
    with open(args.output_json, "r") as f:
        result = json.load(f)
    print(f"      Resume mode: {len(result)} already done, skipping.")

todo = [r for r in relations if r not in result]
print(f"      To embed: {len(todo)} relations.")

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
    for rel, emb in zip(batch, embeddings):
        result[rel] = emb
    # checkpoint every 5000
    if (i // args.batch_size) % 50 == 49:
        with open(args.output_json, "w") as f:
            json.dump(result, f)
        print(f"  [checkpoint] {len(result)}/{len(relations)} saved.")

# ── save final ────────────────────────────────────────────────────────────────
print(f"[3/3] Saving to {args.output_json} ...")
with open(args.output_json, "w") as f:
    json.dump(result, f)

dims = len(next(iter(result.values())))
print(f"\n✅ Done. {len(result)} relations embedded, dim={dims}.")
print(f"   Output: {args.output_json}")