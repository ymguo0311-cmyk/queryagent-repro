from SPARQLWrapper import SPARQLWrapper, JSON

sparql = SPARQLWrapper("https://dbpedia.org/sparql")

# 测试1: DBpedia自己的数据
query1 = """
SELECT ?s WHERE {
  ?s a <http://dbpedia.org/ontology/Person>
} LIMIT 5
"""

print("Test 1: DBpedia native query")
sparql.setQuery(query1)
sparql.setReturnFormat(JSON)
results = sparql.query().convert()
print(f"Results: {len(results['results']['bindings'])}")
print(f"Sample: {results['results']['bindings'][0] if results['results']['bindings'] else 'None'}")

# 测试2: Freebase格式查询
query2 = """
PREFIX : <http://rdf.freebase.com/ns/>
SELECT ?relation WHERE {
  :m.0m_sb ?relation ?x
} LIMIT 10
"""

print("\nTest 2: Freebase MID query")
sparql.setQuery(query2)
sparql.setReturnFormat(JSON)
results = sparql.query().convert()
print(f"Results: {len(results['results']['bindings'])}")
print(f"Sample: {results['results']['bindings'][0] if results['results']['bindings'] else 'None'}")

# 测试3: 查询DBpedia是否有Freebase链接
query3 = """
SELECT ?s ?freebase WHERE {
  ?s <http://www.w3.org/2002/07/owl#sameAs> ?freebase .
  FILTER(CONTAINS(STR(?freebase), "freebase"))
} LIMIT 5
"""

print("\nTest 3: DBpedia-Freebase links")
sparql.setQuery(query3)
sparql.setReturnFormat(JSON)
results = sparql.query().convert()
print(f"Results: {len(results['results']['bindings'])}")
if results['results']['bindings']:
    print(f"Sample: {results['results']['bindings'][0]}")
