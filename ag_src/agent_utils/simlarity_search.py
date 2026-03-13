# -*- coding: utf-8 -*-
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from agent_utils.config import *
from sentence_transformers import SentenceTransformer, util
from agent_utils.config import NEO4J_DRIVER


use_embedding = config['use_embedding']
openai_embedding = config['openai_embedding']
sentenceTransformer_embedding = config['sentence_transformer']

if sentenceTransformer_embedding:
    transformer_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')


def _get_query_embedding(query):
    """
    Embed query using text-embedding-3-small (1536d).
    Same model used to pre-compute fb_relation_embed_1536.json.
    """
    from openai import OpenAI
    client = OpenAI(api_key=all_key[0], base_url=config['api_base'])
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=[query])
    return resp.data[0].embedding  # 1536d list


def faiss_filter_with_udp(query, relation_list, entity_mid=None):
    """
    Semantic relation ranking via Neo4j UDP (text-embedding-3-small, 1536d).

    Mode 1 — entity-constrained:
        CALL semantic.matchEdgePathForEntity(entityMid, queryEmb, topK)
        Only considers relations connected to the given entity.
        Requires Freebase entity graph to be ingested into Neo4j.

    Mode 2 — global fallback:
        CALL semantic.matchEdgePath(queryEmb, topK)
        Searches all 24K :Relation nodes.
        Used when entity not found in Neo4j (Freebase not yet ingested).
    """
    query_emb = _get_query_embedding(query)

    # Mode 1: entity-constrained
    if entity_mid:
        try:
            with NEO4J_DRIVER.session() as session:
                result = session.run("""
                    CALL semantic.matchEdgePathForEntity($mid, $emb, $topK)
                """, mid=entity_mid, emb=query_emb, topK=10)
                relations = [row["relationName"] for row in result]

            if relations:
                print(f"[UDP] Entity-constrained: {entity_mid}, top-3: {relations[:3]}")
                return relations
            else:
                print(f"[UDP] Entity {entity_mid} not in Neo4j, falling back to global search")

        except Exception as e:
            print(f"[UDP] matchEdgePathForEntity error: {e}, falling back")

    # Mode 2: global fallback
    with NEO4J_DRIVER.session() as session:
        result = session.run("""
            CALL semantic.matchEdgePath($emb, $topK)
        """, emb=query_emb, topK=10)
        relations = [row["relationName"] for row in result]

    print(f"[UDP] Global ranking, top-3: {relations[:3]}")
    return relations


def faiss_filter(query, relation_list, entity_mid=None):
    """
    Main entry point for relation ranking.

    use_neo4j=True  → UDP semantic ranking (Neo4j vector index)
    use_neo4j=False → OpenAI embedding cosine similarity (baseline pro)
    """
    if config.get('use_neo4j', False):
        return faiss_filter_with_udp(query, relation_list, entity_mid=entity_mid)

    # Baseline pro: OpenAI embedding cosine similarity
    if use_embedding:
        if openai_embedding:
            if query in q_embedding_map.keys():
                question_embedding = q_embedding_map[query] #input query
            else:
                raise KeyError(f"Missing question embedding for: {query[:120]}")

            relation_list = relation_list[:300]
            relation_embeddings = []
            for rel in relation_list:
                if rel in r_embedding_map.keys(): # shortlisted relations
                    relation_embeddings.append(r_embedding_map[rel])
                else:
                    raise KeyError(f"Missing relation embedding for: {rel}")

        if sentenceTransformer_embedding:
            question_embedding = transformer_model.encode(query, convert_to_tensor=True)
            relation_embeddings = transformer_model.encode(relation_list, convert_to_tensor=True)

        similarities = util.pytorch_cos_sim(question_embedding, relation_embeddings)
        sorted_relations = sorted(
            zip(relation_list, similarities.tolist()[0]),
            key=lambda x: x[1], reverse=True
        )
        return [rel for rel, _ in sorted_relations]