# 在本地Mac上跑这个测试
from neo4j import GraphDatabase
from openai import OpenAI
import os

driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "Gym070311"))
client = OpenAI(api_key=OPEN_AI_KEY)

# 生成一个1536d的query embedding
resp = client.embeddings.create(model="text-embedding-3-small", input=["which red dwarf star has the lowest temperature"])
query_emb = resp.data[0].embedding  # 1536d

print(f"embedding dim: {len(query_emb)}")

# 直接调Java UDP
with driver.session() as session:
    try:
        result = session.run("""
            CALL semantic.matchEdgePath($emb, 10)
           
        """, emb=query_emb)
        rows = list(result)
        print(f"✅ UDP works! Got {len(rows)} results")
        for r in rows[:3]:
            print(f"  {r['relationName']} ({r['score']:.4f})")
    except Exception as e:
        print(f"❌ UDP failed: {e}")

driver.close()