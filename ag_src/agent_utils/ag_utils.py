# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name : ag_utils.py 
   Description :  utils func
   Author :       HX / Modified for Neo4j backend
-------------------------------------------------
   CHANGES:
   1. execute_sparql() -> Mock mode, returns structured fake data
   2. id2label() -> Mock mode, no longer calls SPARQLWrapper
   3. NEO4J_DRIVER -> kept but wrapped in try/except (won't crash if Neo4j offline)
   4. execute_cypher_with_udp() -> kept for future Neo4j integration
-------------------------------------------------
"""
from SPARQLWrapper import SPARQLWrapper, JSON
import os
import re
import json
import pandas as pd
import numpy as np
from agent_utils.config import *
from tqdm import tqdm
import tiktoken
import openai
import requests


# ── Neo4j driver (optional, won't crash if Neo4j not running) ──────────────
try:
    from neo4j import GraphDatabase
    NEO4J_DRIVER = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "Gym070311"))
    print("[INFO] Neo4j driver connected.")
except Exception as _neo4j_err:
    NEO4J_DRIVER = None
    print(f"[WARN] Neo4j not available: {_neo4j_err}. Running in Mock mode.")

# ── SPARQL template (kept for reference, not used in mock mode) ────────────
sparql_id_fb = """PREFIX : <http://rdf.freebase.com/ns/>\nSELECT DISTINCT ?tailEntity\nWHERE {\n  ?entity :type.object.name ?tailEntity .\n    FILTER(?entity = :%s)  \n}"""

# ── Mock data: realistic Freebase-style relations for GrailQA ─────────────
MOCK_RELATIONS = {
    "forward": [
        "people.person.date_of_birth",
        "people.person.place_of_birth",
        "people.person.nationality",
        "people.person.profession",
        "organization.organization.founders",
        "film.film.directed_by",
        "film.film.starring",
        "music.artist.genre",
        "sports.sports_team.sport",
        "location.location.containedby",
    ],
    "backward": [
        "people.person.employment_history",
        "organization.organization.leadership",
        "film.film.production_companies",
        "music.album.artist",
        "sports.sports_team.founded",
    ]
}

MOCK_QUERY_RESULTS = [
    {"tailEntity": "http://rdf.freebase.com/ns/m.mock_entity_001"},
    {"tailEntity": "http://rdf.freebase.com/ns/m.mock_entity_002"},
]

# ─────────────────────────────────────────────────────────────────────────────

def clean_str(p):
    return p.encode().decode("unicode-escape").encode("latin1").decode("utf-8")


def clean_para(para):
    if type(para) == str:
        para = para.strip()
        if len(para) != 0:
            if 'XMLSchema#' in para:
                if para[-1] != '"':
                    para = '"' + para[1:]
                if para[-1] == '"' or para[-1] == "'":
                    para = para[:-1]
            else:
                if para[0] == '"' or para[0] == "'":
                    para = para[1:]
                if para[-1] == '"' or para[-1] == "'":
                    para = para[:-1]
    return para


def num_tokens_from_string(string: str, model_name: str = "gpt-3.5-turbo") -> int:
    encoding = tiktoken.encoding_for_model(model_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def readjson(path):
    with open(path, mode='r', encoding='utf-8') as load_f:
        data_ = json.load(load_f)
    return data_


class Dict2Obj(dict):
    def __getattr__(self, key):
        value = self.get(key)
        return Dict(value) if isinstance(value, dict) else value

    def __setattr__(self, key, value):
        self[key] = value


class IntEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, int):
            return int(obj)
        return super().default(obj)


def savejson(file_name, json_info, indent=4):
    with open('{}.json'.format(file_name), 'w') as fp:
        json.dump(json_info, fp, indent=indent, sort_keys=False, cls=IntEncoder)


def print_rgb(r, g, b, input_str):
    custom_color = '\033[38;2;' + str(r) + ';' + str(g) + ';' + str(b) + 'm' + input_str + '\033[0m'
    print(custom_color)


def print_thought(*input_str):
    print_rgb(82, 121, 186, ' '.join(input_str))


def print_action(*input_str):
    print_rgb(90, 187, 147, ' '.join(input_str))


def print_obs(*input_str):
    print_rgb(141, 98, 157, ' '.join(input_str))


def print_error(*input_str):
    print_rgb(255, 105, 97, ' '.join(input_str))


def print_refine(*input_str):
    print_rgb(255, 200, 0, ' '.join(input_str))


def f1_score(pred, golden):
    if len(pred) == 0 and len(golden) == 0:
        f1 = 1
    elif len(pred) == 0 and len(golden) != 0:
        f1 = 0
    elif len(pred) != 0 and len(golden) == 0:
        f1 = 0
    else:
        p = len([x for x in pred if x in golden]) / len(pred)
        r = len([x for x in golden if x in pred]) / len(golden)
        if p == 0 or r == 0:
            f1 = 0
        else:
            f1 = 2 * p * r / (p + r)
    return f1


def llm(prompt, model_name, stop=["\n"]):
    got_result = False
    while not got_result:
        try:
            current_key = all_key[0]
            del all_key[0]
            all_key.append(current_key)

            from openai import OpenAI
            client = OpenAI(
                api_key=current_key,
                base_url=config['api_base']
            )
            response = client.chat.completions.create(
                model=config['model'],
                messages=[{'role': 'user', 'content': prompt}],
                temperature=0,
                max_tokens=256,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                stop=stop
            )
            response = response.choices[0].message.content.strip()
            got_result = True

        except Exception as e:
            if 'You exceeded your current quota' in str(e):
                print('bad key: ', current_key)
            elif "maximum context length" in str(e) or "Request too large" in str(e):
                print(e)
                return PROMPT_TOO_LONG_ERROR
            else:
                print(e)

    return response

def abandon_rels(relation):
    if config['dataset'] == 'webqsp':
        if (relation == "type.object.type" or relation == "type.object.name"
                or relation.startswith("common.") or relation.startswith("freebase.")
                or "sameAs" in relation):
            return True
    elif config['dataset'] == 'grailqa':
        if (relation.startswith("type.") or relation.startswith("common.") or relation.startswith("freebase.")):
            return True
    elif config['dataset'] == 'graphq':
        if (relation == "type.object.type" or relation == "type.object.name"
                or relation.startswith("common.") or relation.startswith("freebase.")
                or "sameAs" in relation):
            return True


def replace_relation_prefix(relations):
    return [relation['relation'].replace("http://rdf.freebase.com/ns/", "") for relation in relations]


def replace_entities_prefix(entities):
    return [entity['tailEntity'].replace("http://rdf.freebase.com/ns/", "") for entity in entities]


def table_result_to_list(res):
    if len(res) == 0:
        return {}
    else:
        key_list = res[0].keys()
        result = {}
        for key in key_list:
            result[key] = list(set([item[key] for item in res]))
        return result


def execute_sql(sql_txt):
    res = conn.query(sql_txt).all()
    res = [x.as_dict() for x in res]
    return res


def execute_cypher_with_udp(cypher_txt):
    """
    执行 Cypher 查询 (future: 调用 Semantic UDP)
    当 Neo4j 可用时启用。
    """
    if NEO4J_DRIVER is None:
        print_error("[execute_cypher_with_udp] Neo4j not available, returning mock.")
        return MOCK_QUERY_RESULTS

    res = []
    try:
        with NEO4J_DRIVER.session() as session:
            result = session.run(cypher_txt)
            for record in result:
                res.append(record.data())
        return res
    except Exception as e:
        print_error(f"Neo4j Query Error: {e}")
        return []

def execute_sparql(sparql_txt):
    # execute SPARQL
    config['query_cnt'] += 1
    if config['dataset'] in ['grailqa', 'webqsp', 'graphq']:
        sparql_txt = 'PREFIX : <http://rdf.freebase.com/ns/>\n' + sparql_txt

    try:
        sparql = SPARQLWrapper(SPARQLPATH)
        sparql.setQuery(sparql_txt)
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()

        res = []
        for x in results["results"]["bindings"]:
            res_item = {}
            for k, v in x.items():
                res_item[k] = v['value']
            res.append(res_item)
        return res
    except:
        print("SPARQL query error")
        print(sparql_txt)
        return []

# ══════════════════════════════════════════════════════════════════════════════
#  CRITICAL FIX: execute_sparql — Mock Mode
#  inactive SPARQL endpoint (114.212.81.217:8896) 
#  Orginal function: send SPARQL to Virtuoso endpoint，execute query
#  Mock function： block SPARQL，return Mock data
#  After neo4j is ready，replace Mock branch with execute_cypher_with_udp() 
# ══════════════════════════════════════════════════════════════════════════════
'''
def execute_sparql(sparql_txt):
    config['query_cnt'] += 1 # 计数器： 记录Agent调用了多少次查询

    # ── output the of LLM ──────────────────────────
    print_action(f"[SPARQL #{config['query_cnt']}] {sparql_txt[:200]}")

    sparql_upper = sparql_txt.upper()

    #-----test for certain question------
    if '?RELATION' in sparql_upper and 'm.0m_sb' in sparql_txt.lower():
        print("[DEBUG] Using Neo4j for m.0m_sb relations...")
        
        try:
            with NEO4J_DRIVER.session() as session:
                # 查询出边
                forward_result = session.run("""
                    MATCH (e:Entity {mid: "m.0m_sb"})-[r:RELATION]->()
                    RETURN DISTINCT r.type AS relation
                """)
                forward = [{"relation": f"http://rdf.freebase.com/ns/{row['relation']}"} 
                          for row in forward_result]
                
                # 查询入边
                backward_result = session.run("""
                    MATCH ()-[r:RELATION]->(e:Entity {mid: "m.0m_sb"})
                    RETURN DISTINCT r.type AS relation
                """)
                backward = [{"relation": f"http://rdf.freebase.com/ns/{row['relation']}"} 
                           for row in backward_result]
                
                result = forward + backward
                print_obs(f"[Neo4j] Returning {len(result)} relations from Neo4j")
                return result
        except Exception as e:
            print_error(f"[Neo4j Error] {e}, falling back to Mock")
    # get_relation 查询 (SELECT ?relation)
    if '?RELATION' in sparql_upper:
        mock_relations = (
            [{"relation": f"http://rdf.freebase.com/ns/{r}"}
            for r in MOCK_RELATIONS["forward"]]
            +
            [{"relation": f"http://rdf.freebase.com/ns/{r}"}
             for r in MOCK_RELATIONS["backward"]]
        )
        print_obs(f"[Mock] Returning {len(mock_relations)} relations.")
        return mock_relations
        
    # id2label 查询 (SELECT ?tailEntity)
    if '?TAILENTITY' in sparql_upper:
        print_obs("[Mock] Returning mock tailEntity.")
        return [{"tailEntity": "mock_entity_label"}]

    # execute / add_fact 查询 (general SELECT)v
    print_obs("[Mock] Returning mock query result.")
    return MOCK_QUERY_RESULTS
'''

# ══════════════════════════════════════════════════════════════════════════════
#  CRITICAL FIX: id2label — Mock Mode
#   Original function: transform Freebase ID to readable name
#   Current Function: return entity_id itself as label. Replace by neo4j query afterwards
#  原来直接调 SPARQLWrapper，endpoint 死了会在这里崩。
# ══════════════════════════════════════════════════════════════════════════════
def id2label(entity_id):
    
    # 如果是 mock entity，给一个可读的假名字
    if entity_id == "UnName_Entity" or entity_id is None:
        return "UnName_Entity"
    if "mock" in str(entity_id).lower():
        return f"MockLabel({entity_id})"
    # 对真实 mid 也直接返回 id（不会崩，但没有真实 label）
    return entity_id


def get_brief_obs(obs):
    brief_obs = obs
    if type(obs) == dict:
        brief_obs = {}
        for k, v in obs.items():
            if type(v) == list:
                brief_obs[k] = v[:3] + ['...'] if len(v) > 3 else v
            elif type(v) == dict:
                brief_obs[k] = dict(list(v.items())[:3]) if len(v) > 3 else v
    elif type(obs) == str:
        if "The value of variable" in obs:
            match_res = re.findall('The value of variable (\?.*?) is (\[.*?\])', obs)
            brief_obs = ""
            for tur in match_res:
                res_list = eval(tur[1])
                if len(res_list) <= 2:
                    brief_obs += 'The value of variable ' + tur[0] + ' is ' + tur[1] + '. '
                else:
                    brief_obs += 'The value of variable ' + tur[0] + ' is ' + str(
                        eval(tur[1])[:2] + (['...'])) + '. '
    return str(brief_obs)


def get_dynamic_history(solve_history):
    if config['dataset'] == 'metaqa':
        dynamic_history = solve_history['base_prompt'] + "\nQuestion: " + solve_history[
            'question'] + " \nEntity: " + str(solve_history['entity']) + "\nRelation for the entity: " + str(
            solve_history['initial_rel']) + "\n"
    elif config['dataset'] == 'wikisql':
        dynamic_history = solve_history['base_prompt'] + "\nQuestion: " + solve_history[
            'question'] + " \nTable Header: " + str(solve_history['header']) + "\n"
    elif config['dataset'] == 'wtq':
        dynamic_history = solve_history['base_prompt'] + "\nQuestion: " + solve_history[
            'question'] + " \nTable Header: " + str(solve_history['header']) + "\n"
    else:
        dynamic_history = solve_history['base_prompt'] + "\nQuestion: " + solve_history[
            'question'] + " \nEntity: " + str(solve_history['entity']) + "\n"

    for index, his in enumerate(solve_history['TAO_list']):
        if index < len(solve_history['TAO_list']) - 2:
            obs = "It's not important. Focus on recent observations."
            dynamic_history += f"Thought {index + 1}: {his['Thought']}\nAction {index + 1}: {his['Action']}\nObservation {index + 1}: {obs}\n"
        elif index == len(solve_history['TAO_list']) - 2:
            obs = str(his['Observation'])
            if 'ERROR_IN_STEP' in obs and len(obs) > 250:
                obs = "It's not important. Focus on recent observations."
            dynamic_history += f"Thought {index + 1}: {his['Thought']}\nAction {index + 1}: {his['Action']}\nObservation {index + 1}: {obs}\n"
        else:
            dynamic_history += f"Thought {index + 1}: {his['Thought']}\nAction {index + 1}: {his['Action']}\nObservation {index + 1}: {str(his['Observation'])}\n"

    dynamic_history = dynamic_history + f"Thought {len(solve_history['TAO_list']) + 1}:"
    return dynamic_history


def try_step(env, action):
    attempts = 0
    while attempts < 10:
        try:
            return env.step(action)
        except requests.exceptions.Timeout:
            attempts += 1


def process_webqsp():
    pyql_file = readjson('../../data/webqsp_test_pyql.json')
    webqsp_train = readjson('/home2/xhuang/PycharmProject/GLLLM/ReAct/data/WebQSP/WebQSP.test.json')['Questions']
    processed_examples = list()

    for example in tqdm(webqsp_train):
        parse = example["Parses"][0]
        grounded_items = {}
        if "TopicEntityMid" in parse and parse["TopicEntityMid"]:
            grounded_items.update({parse["TopicEntityName"]: parse["TopicEntityMid"]})
        if "Constraints" in parse:
            for cons in parse["Constraints"]:
                if cons["ArgumentType"] == "Entity":
                    grounded_items.update({cons["EntityName"]: cons["Argument"]})
                elif cons["ArgumentType"] == "Value":
                    if cons["ValueType"] == "String":
                        mid = f'"{cons["Argument"]}"'
                        grounded_items.update({mid: f'{mid}@en'})
                    elif cons["ValueType"] == "DateTime":
                        mid = f'"{cons["Argument"]}"^^<http://www.w3.org/2001/XMLSchema#dateTime>'
                        grounded_items.update({cons["Argument"]: mid})
                    else:
                        raise Exception(f'cons: {cons}; example id: {example["QuestionId"]}')
                else:
                    raise Exception(f'cons: {cons}; example id: {example["QuestionId"]}')

        different_answer = []
        for par in example['Parses']:
            exe_res = table_result_to_list(execute_sparql(par['Sparql']))
            if exe_res == []:
                exe_res = []
            else:
                if len(exe_res.keys()) > 1:
                    print(example['QuestionId'], ' multi answer key')
                answer_key = list(exe_res.keys())[0]
                exe_res = exe_res[answer_key]
                exe_res = [x.replace("http://rdf.freebase.com/ns/", "") for x in exe_res]
            different_answer.append(exe_res)

        processed_examples.append({
            "qid": example["QuestionId"],
            "question": example["ProcessedQuestion"],
            "answer": different_answer[0],
            "different_answer": different_answer,
            "entity_linking": grounded_items,
            "sparql_query": parse["Sparql"],
            "PyQL": [x['pyql'] for x in pyql_file if x['ID'] == example['QuestionId']][0]
        })

    savejson('/home2/xhuang/PycharmProject/GLLLM/ReAct/data/WebQSP/WebQSP_test_processed', processed_examples)