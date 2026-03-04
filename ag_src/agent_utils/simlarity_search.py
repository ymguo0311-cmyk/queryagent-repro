# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name : simlarity_search.py 
   Description :  
   Author :       HX
   date :    2023/12/5 15:28 
-------------------------------------------------
"""
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
#from pyserini.search import FaissSearcher, LuceneSearcher
#from pyserini.search.hybrid import HybridSearcher
#from pyserini.search.faiss import AutoQueryEncoder
from agent_utils.config import *
import pandas as pd
import json
import sys
from sentence_transformers import SentenceTransformer, util
from agent_utils.config import NEO4J_DRIVER

_udp_model = None




use_embedding = config['use_embedding']
openai_embedding = config['openai_embedding']
sentenceTransformer_embedding = config['sentence_transformer']

if sentenceTransformer_embedding:
    # load SentenceTransformer model
    transformer_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
#if not use_embedding:
#    query_encoder = AutoQueryEncoder(encoder_dir='facebook/contriever', pooling='mean')
#
#    bm25_searcher = LuceneSearcher('../../data/contriever_fb_relation/index_relation_fb')
#    contriever_searcher = FaissSearcher('../../data/contriever_fb_relation/freebase_contriever_index', query_encoder)

#    hsearcher = HybridSearcher(contriever_searcher, bm25_searcher)


def get_openai_embedding(input_message):
    import openai

    ok = False
    while not ok:
        try:
            current_key = all_key[0]
            del all_key[0]
            all_key.append(current_key)
            openai.api_key = current_key
            response = openai.Embedding.create(engine="text-embedding-ada-002",
                                               input=input_message)
            ok = True
        except Exception as e:
            if 'You exceeded your current quota' in str(e) or 'The OpenAI account associated with this API key' in str(e):
                print('bad key: ', current_key)
            else:
                print(e)
        print('stuck in here get_openai_embedding')

    return response['data']

# similarity_search.py

from sentence_transformers import SentenceTransformer
from agent_utils.config import NEO4J_DRIVER

# ← 在文件顶部定义全局变量
_udp_model = None

def faiss_filter_with_udp(query, relation_list):
    global _udp_model
    
    # 只加载一次模型
    if _udp_model is None:
        print("[INFO] Loading sentence-transformers model for UDP...")
        _udp_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    
    print(f"\n[DEBUG faiss_filter_with_udp] Query: {query[:50]}")
    print(f"[DEBUG] Query embedding dims calculation...")
    
    # 编码问题
    query_emb = _udp_model.encode(query, normalize_embeddings=True).tolist()
    print(f"[DEBUG] Query embedding dims: {len(query_emb)}")
    print(f"[DEBUG] Relation list size: {len(relation_list)}")
    
    # 调用UDP
    with NEO4J_DRIVER.session() as session:
        result = session.run("""
            CALL semantic.matchEdgePath($emb, 40)
            YIELD relationName, score
            RETURN relationName, score
            ORDER BY score DESC
        """, emb=query_emb)
        
        ranked_with_scores = [(row['relationName'], row['score']) for row in result]
    
    # 打印排序结果
    print(f"\n[UDP Ranking] Top 10 relations:")
    for rel, score in ranked_with_scores[:10]:
        print(f"  {rel:.50s}  score={score:.4f}")
    
    # 过滤到relation_list里
    ranked = [rel for rel, _ in ranked_with_scores]
    filtered = [r for r in ranked if r in relation_list]
    remaining = [r for r in relation_list if r not in filtered]
    
    return (filtered + remaining)[:len(relation_list)]


def faiss_filter(query, relation_list):
    print(f"\n[DEBUG faiss_filter] use_neo4j = {config.get('use_neo4j', False)}")
    print(f"[DEBUG faiss_filter] query = {query[:50]}")
    print(f"[DEBUG faiss_filter] relation_list length = {len(relation_list)}")
    
    if config.get('use_neo4j', False):
        print("[DEBUG] Calling UDP version...")
        return faiss_filter_with_udp(query, relation_list)
 
    
    print("[DEBUG] Using original BM25 version...")
    
    if use_embedding:
        if openai_embedding:
            if query in q_embedding_map.keys():
                question_embedding = q_embedding_map[query]
            else:
                raise KeyError(f"Missing question embedding for: {query[:120]}")

            relation_list = relation_list[:300]
            print('the len of relation list in openai embedding:', len(relation_list))

            relation_embeddings = []
            for rel in relation_list:
                if rel in r_embedding_map.keys():
                    relation_embeddings.append(r_embedding_map[rel])
                else:
                    raise KeyError(f"Missing relation embedding for: {rel}")
                if type(relation_embeddings[-1])!=list:
                    pass
        # sentenceTransformer embedding
        if sentenceTransformer_embedding:
            question_embedding = transformer_model.encode(query, convert_to_tensor=True)
            relation_embeddings = transformer_model.encode(relation_list, convert_to_tensor=True)

        # Calculate the similarity of question and relations
        similarities = util.pytorch_cos_sim(question_embedding, relation_embeddings)

        sorted_relations = [(relation, score) for relation, score in zip(relation_list, similarities.tolist()[0])]
        sorted_relations = sorted(sorted_relations, key=lambda x: x[1], reverse=True)

        # return sorted relation list, not include score
        sorted_relation_list = [relation[0] for relation in sorted_relations]
        return sorted_relation_list
    else:
        query_encoding = contriever_searcher.query_encoder.encode(query)
        hits = []
        relation_list = relation_list[:300]

        for lines in relation_list:
            query_token = lines.replace(".", " ").replace("_", " ").strip()
            hit = bm25_searcher.search(query_token, k=1)[0]
            # hit = hsearcher.search(query_token, k=1)[0]
            hits.append(hit)
        retrieved_paragraphs_ids = [hit.docid for hit in hits]

        retrieved_paragraphs_embeddings = []
        for doc_id in retrieved_paragraphs_ids:
            # get embedding by id
            embedding = contriever_searcher.index.reconstruct(int(doc_id))
            retrieved_paragraphs_embeddings.append(embedding)

        similarity_scores = cosine_similarity(query_encoding.reshape(1, -1), np.array(retrieved_paragraphs_embeddings))
        similarity_scores = list(similarity_scores[0])

        df = pd.DataFrame({'relation': relation_list, 'score': similarity_scores})
        df = df.sort_values(by='score', ascending=False)
        # print(df)
        return list(df['relation'])

