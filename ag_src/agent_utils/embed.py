"""
regenerate_embeddings_768.py
─────────────────────────────
把fb_relation.txt重新编码成768维向量（使用all-mpnet-base-v2）

用法：
    python regenerate_embeddings_768.py
"""

import json
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# 加载模型（768维）
print("[1/3] Loading SentenceTransformer model (768-dim)...")
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
print(f"      Model loaded, embedding dim = {model.get_sentence_embedding_dimension()}")

# 读取关系列表
print("[2/3] Reading fb_relation.txt...")
with open('~/Desktop/QueryAgent/data/fb_relation.txt'.replace('~', '/Users/yammmy'), 'r') as f:
    relations = [line.strip() for line in f if line.strip()]

print(f"      Found {len(relations)} relations")

# 编码
print("[3/3] Encoding relations...")
embeddings = {}

batch_size = 100
for i in tqdm(range(0, len(relations), batch_size)):
    batch = relations[i:i+batch_size]
    batch_embeddings = model.encode(batch, normalize_embeddings=True)
    
    for rel, emb in zip(batch, batch_embeddings):
        embeddings[rel] = emb.tolist()

# 保存
output_path = '/Users/yammmy/Desktop/QueryAgent/data/fb_relation_embed_768.json'
print(f"\n[4/4] Saving to {output_path}...")
with open(output_path, 'w') as f:
    json.dump(embeddings, f)

print(f"\nDone! Generated {len(embeddings)} embeddings (768-dim)")
print(f"Output: {output_path}")