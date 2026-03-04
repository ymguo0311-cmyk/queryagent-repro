"""
ingest_relations.py
───────────────────
import fb_relation_embed.json to Neo4j+ build Vector Index。


    python ingest_relations.py \
        --json_path ../../data/fb_relation_embed.json \
        --neo4j_uri bolt://localhost:7687 \
        --neo4j_user neo4j \
        --neo4j_password password

(fb_relation_embed.json):
    {
        "transport.transit_line.terminus": [0.12, 0.34, ...],
        "people.person.date_of_birth":     [0.56, 0.78, ...],
        ...
    }
"""

import json
import argparse
from neo4j import GraphDatabase
from tqdm import tqdm

# ── parameter ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--json_path",      default="../../data/fb_relation_embed.json")
parser.add_argument("--neo4j_uri",      default="bolt://localhost:7687")
parser.add_argument("--neo4j_user",     default="neo4j")
parser.add_argument("--neo4j_password", default="password")
parser.add_argument("--batch_size",     type=int, default=500)
args = parser.parse_args()

# ── connect Neo4j ───────────────────────────────────────────────────────────
driver = GraphDatabase.driver(
    args.neo4j_uri,
    auth=(args.neo4j_user, args.neo4j_password)
)

# ── read embedding file ───────────────────────────────────────────────────
print(f"[1/4] Loading embeddings from {args.json_path} ...")
with open(args.json_path, "r") as f:
    data = json.load(f)

items     = list(data.items())
total     = len(items)
dims      = len(items[0][1])
print(f"      Found {total} relations, embedding dim = {dims}")

# ── build Vector Index ───────────────────────────────────────────────────────
print("[2/4] Creating vector index 'relation_index' ...")
with driver.session() as session:
    # clean old index if exist
    try:
        session.run("DROP INDEX relation_index IF EXISTS")
    except Exception:
        pass

    session.run(f"""
        CREATE VECTOR INDEX relation_index IF NOT EXISTS
        FOR (r:Relation) ON r.embedding
        OPTIONS {{
            indexConfig: {{
                `vector.dimensions`: {dims},
                `vector.similarity_function`: 'cosine'
            }}
        }}
    """)
    print("      Vector index created.")

# ── import batched node ──────────────────────────────────────────────────────────
print(f"[3/4] Ingesting {total} Relation nodes (batch_size={args.batch_size}) ...")

def ingest_batch(session, batch):
    session.run("""
        UNWIND $batch AS item
        MERGE (r:Relation {name: item.name})
        SET r.embedding = item.embedding
    """, batch=[{"name": k, "embedding": v} for k, v in batch])

with driver.session() as session:
    for i in tqdm(range(0, total, args.batch_size)):
        batch = items[i : i + args.batch_size]
        ingest_batch(session, batch)

# ── verification ──────────────────────────────────────────────────────────────────
print("[4/4] Verifying ...")
with driver.session() as session:
    count = session.run(
        "MATCH (r:Relation) RETURN count(r) AS cnt"
    ).single()["cnt"]
    print(f"      Neo4j now has {count} Relation nodes.")

    # test vector query
    sample_emb = items[0][1]
    result = session.run("""
        CALL db.index.vector.queryNodes('relation_index', 5, $emb)
        YIELD node, score
        RETURN node.name AS name, score
    """, emb=sample_emb)
    print("      Top-5 nearest neighbors for first relation:")
    for row in result:
        print(f"        {row['name']:.60s}  score={row['score']:.4f}")

driver.close()
print("\nDone! Relations imported successfully.")
