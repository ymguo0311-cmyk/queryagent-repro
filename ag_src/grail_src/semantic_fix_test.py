"""
semantic_fix_test.py
────────────────────
Compares accuracy of:
  1) Our method  = func_list SEMANTIC Cypher (via semantic.matchSubgraph UDP)
  2) QueryAgent   = func_list NON-SEMANTIC (standard) Cypher
both evaluated against the benchmark ground-truth answers (gt_answer).

Golden PyQL queries are NOT run (GTs are taken directly from the benchmark).

Run on zxcpu11:
    python3 semantic_fix_test.py
"""

import sys
sys.path.insert(0, '.')

from pyql_to_cypher import PyQLToCypher
from standard_to_semantic import transform_cypher_query

NEO4J_URI      = "neo4j://localhost:7690"
NEO4J_USER     = "neo4j"
NEO4J_PASSWORD = "12345678"

# ── Fail cases using func_list (LLM-generated PyQL) ──────────────────────────

FAIL_CASES = [
    {
        "index": 13,
        "question": "The developer of Battle Forge also developed what video game version?",
        "standard_pred": ["m.03d8ndd", "m.0hr7ks5", "m.0f4081", "m.0fbc03"],
        "gt_answer": ["m.05nspt0", "m.0kyq3h2", "m.0kyq5k0", "m.0kz27qj"],
        "root_cause": "LLM used cvg.cvg_developer.games_developed (finds videogames) instead of cvg.game_version.developer (finds game versions). Wrong relation path.",
        "func_list": [
            "add_fact(m.04f56nl, base.wikipedia_infobox.video_game.developer, ?developer)",
            "add_fact(?developer, cvg.cvg_developer.games_developed, ?game)",
            "add_filter(?game, !=, m.04f56nl)",
            "set_answer(?game)",
        ],
        "golden_pyql": [
            "add_type_constrain(cvg.game_version, ?x0)",
            "add_fact(m.04f56nl, cvg.computer_videogame.developer, ?x1)",
            "add_fact(?x0, cvg.game_version.developer, ?x1)",
            "set_answer(?x0)",
        ],
    },
    {
        "index": 15,
        "question": "What channel access method has a child method of a channel access method with a parent method of packet mode multiple access?",
        "standard_pred": ["m.05y8dl3"],
        "gt_answer": ["m.012vrj", "m.026lnjl", "m.0h0fwx"],
        "root_cause": "LLM traversed path in wrong order. Should find x1 with parent=m.05y8dl3, then x0 as child of x1.",
        "func_list": [
            "add_fact(m.05y8dl3, engineering.channel_access_method.child_method, ?child)",
            "add_fact(?child, engineering.channel_access_method.parent_method, ?method)",
            "set_answer(?method)",
        ],
        "golden_pyql": [
            "add_type_constrain(engineering.channel_access_method, ?x0)",
            "add_fact(?x1, engineering.channel_access_method.parent_method, m.05y8dl3)",
            "add_fact(?x1, engineering.channel_access_method.child_method, ?x0)",
            "set_answer(?x0)",
        ],
    },

    {
        "index": 19,
        "question": "Name the video game that has subjects of hammer throwing in it.",
        "standard_pred": ["g.125_kjjdb"],
        "gt_answer": ["m.02q8x4x"],
        "root_cause": "LLM added unnecessary base.schemastaging.context_name.pronunciation step.",
        "func_list": [
            "add_fact(m.0byp9, cvg.computer_game_subject.games, ?game)",
            "add_fact(?game, base.schemastaging.context_name.pronunciation, ?name)",
            "set_answer(?game)",
        ],
        "golden_pyql": [
            "add_type_constrain(cvg.computer_videogame, ?x0)",
            "add_fact(?x0, cvg.computer_videogame.subjects, m.0byp9)",
            "set_answer(?x0)",
        ],
    },
    {
        "index": 22,
        "question": "For cameras with camera iso capability high iso auto, what camera image stabilization type is used?",
        "standard_pred": ["m.01vspn7", "m.022q2pk"],
        "gt_answer": ["m.022q2pk"],
        "root_cause": "LLM used digicams.digital_camera.image_stabilization returning both nodes. Golden PyQL uses inverse relation digicams.image_stabilization_type.digital_camera.",
        "func_list": [
            "add_fact(m.02nqg65, digicams.camera_iso.cameras, ?camera)",
            "add_fact(?camera, digicams.digital_camera.image_stabilization, ?stabilization_type)",
            "set_answer(?stabilization_type)",
        ],
        "golden_pyql": [
            "add_type_constrain(digicams.image_stabilization_type, ?x0)",
            "add_fact(m.02nqg65, digicams.camera_iso.cameras, ?x1)",
            "add_fact(?x0, digicams.image_stabilization_type.digital_camera, ?x1)",
            "set_answer(?x0)",
        ],
    },
    {
        "index": 28,
        "question": "Please explain the unit of viscosity in the international system of units?",
        "standard_pred": ["1.0"],
        "gt_answer": ["m.02sj4qd"],
        "root_cause": "LLM used dimension.units -> viscosity_in_pascal_seconds, returning numeric value. Should use viscosity_unit.measurement_system.",
        "func_list": [
            "add_fact(m.0b2sz, measurement_unit.dimension.units, ?unit)",
            "add_fact(?unit, measurement_unit.viscosity_unit.viscosity_in_pascal_seconds, ?viscosity)",
            "set_answer(?viscosity)",
        ],
        "golden_pyql": [
            "add_type_constrain(measurement_unit.viscosity_unit, ?x0)",
            "add_fact(?x0, measurement_unit.viscosity_unit.measurement_system, m.0c13h)",
            "set_answer(?x0)",
        ],
    },
    {
        "index": 29,
        "question": "What is the organism classification for the fossil Bodo cranium?",
        "standard_pred": ["Homo"],
        "gt_answer": ["m.02gj4w"],
        "root_cause": "LLM used fossil_specimen.organism -> scientific_name, returning name string. Should use organism_classification.fossil_specimens.",
        "func_list": [
            "add_fact(m.0n8_wf9, biology.fossil_specimen.organism, ?organism)",
            "add_fact(?organism, biology.organism_classification.scientific_name, ?classification)",
            "set_answer(?classification)",
        ],
        "golden_pyql": [
            "add_type_constrain(biology.organism_classification, ?x0)",
            "add_fact(?x0, biology.organism_classification.fossil_specimens, m.0n8_wf9)",
            "set_answer(?x0)",
        ],
    },
    {
        "index": 32,
        "question": "What video game series also includes a video game expansion that was designed by blizzard games?",
        "standard_pred": ["m.04cxkz2", "m.03czv2b", "m.03bwwx", "m.026wy8d"],
        "gt_answer": ["m.03bwwx"],
        "root_cause": "LLM used cvg.cvg_developer.games_developed (too broad). Golden PyQL uses cvg.computer_videogame.designers (more specific).",
        "func_list": [
            "add_fact(m.01jx9, cvg.cvg_developer.games_developed, ?expansion)",
            "add_fact(?expansion, cvg.computer_videogame.game_series, ?series)",
            "set_answer(?series)",
        ],
        "golden_pyql": [
            "add_type_constrain(cvg.game_series, ?x0)",
            "add_fact(?x1, cvg.computer_videogame.designers, m.01jx9)",
            "add_fact(?x1, cvg.computer_videogame.game_series, ?x0)",
            "set_answer(?x0)",
        ],
    },
    {
        "index": 47,
        "question": "In which mountain range would one be able to find Stob a' Choire Mheadhoin?",
        "standard_pred": ["m.02k211"],
        "gt_answer": ["m.0b1fp"],
        "root_cause": "LLM used geography.mountain.mountain_range (returns mountain range). Golden PyQL uses geography.mountain_listing.mountains (returns listing node).",
        "func_list": [
            "add_fact(m.03m4l4r, geography.mountain.mountain_range, ?mountain_range)",
            "set_answer(?mountain_range)",
        ],
        "golden_pyql": [
            "add_type_constrain(geography.mountain_listing, ?x0)",
            "add_fact(?x0, geography.mountain_listing.mountains, m.03m4l4r)",
            "set_answer(?x0)",
        ],
    },
]

# ── Neo4j runner ──────────────────────────────────────────────────────────────

def run_query(driver, cypher: str):
    with driver.session() as session:
        result = session.run(cypher)
        rows = result.data()
        if not rows:
            return []
        values = []
        for row in rows:
            for v in row.values():
                if hasattr(v, 'get'):
                    values.append(v.get('mid', str(v)))
                else:
                    values.append(str(v))
        return values

def compare(pred, gt):
    return set(str(x).strip() for x in pred) == set(str(x).strip() for x in gt)

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    translator = PyQLToCypher()

    print("Comparing:")
    print("  (1) Our method  = func_list SEMANTIC Cypher (semantic.matchSubgraph UDP)")
    print("  (2) QueryAgent  = func_list NON-SEMANTIC (standard) Cypher")
    print("  both evaluated against benchmark GT answers. Golden queries NOT run.\n")

    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        driver.verify_connectivity()
        print(f"Connected to Neo4j at {NEO4J_URI}\n")
    except Exception as e:
        print(f"[WARN] Neo4j not available: {e}")
        driver = None

    # Tracking
    semantic_correct, standard_correct = [], []
    semantic_wrong, standard_wrong = [], []
    errors = []

    for tc in FAIL_CASES:
        idx = tc["index"]
        print("=" * 72)
        print(f"[{idx}] {tc['question']}")
        print(f"  GT answer: {tc['gt_answer']}")
        print(f"  Root cause: {tc['root_cause']}")
        print("-" * 72)

        # Translate func_list -> standard Cypher
        try:
            std_query = translator.translate(tc["func_list"])
        except Exception as e:
            print(f"  ❌ func_list translation ERROR: {e}\n")
            errors.append(idx)
            continue

        # Build semantic Cypher from the standard one
        try:
            semantic_query = transform_cypher_query(std_query)
        except Exception as e:
            print(f"  ❌ semantic transform ERROR: {e}\n")
            errors.append(idx)
            continue

        print(f"  [Standard / QueryAgent] Cypher:\n{std_query}\n")
        print(f"  [Semantic / Our method] Cypher:\n{semantic_query}\n")

        # ── Run non-semantic (QueryAgent) query ──
        try:
            pred_standard = run_query(driver, std_query)
        except Exception as e:
            print(f"  ❌ Neo4j ERROR (standard): {e}\n")
            pred_standard = None

        # ── Run semantic (our method) query ──
        try:
            pred_semantic = run_query(driver, semantic_query)
        except Exception as e:
            print(f"  ❌ Neo4j ERROR (semantic): {e}\n")
            pred_semantic = None

        if pred_standard is not None:
            print(f"  [QueryAgent / standard] Result: {pred_standard}")
        if pred_semantic is not None:
            print(f"  [Our method / semantic] Result: {pred_semantic}")
        print(f"  GT answer:                       {tc['gt_answer']}")

        # ── Compare both against GT ──
        std_ok = compare(pred_standard, tc["gt_answer"]) if pred_standard is not None else False
        sem_ok = compare(pred_semantic, tc["gt_answer"]) if pred_semantic is not None else False

        if std_ok:
            standard_correct.append(idx)
        else:
            standard_wrong.append(idx)

        if sem_ok:
            semantic_correct.append(idx)
        else:
            semantic_wrong.append(idx)

        print(f"  QueryAgent (standard) {'✅ CORRECT' if std_ok else '❌ WRONG'}")
        print(f"  Our method (semantic) {'✅ CORRECT' if sem_ok else '❌ WRONG'}\n")

    if driver:
        driver.close()

    n = len(FAIL_CASES)
    print("=" * 72)
    print(f"SUMMARY over {n} fail cases (golden queries not run):")
    print(f"  QueryAgent (func_list, non-semantic):")
    print(f"    ✅ correct: {len(standard_correct)}/{n} → {standard_correct}")
    print(f"    ❌ wrong:   {len(standard_wrong)}/{n} → {standard_wrong}")
    print(f"  Our method (func_list, semantic via UDP):")
    print(f"    ✅ correct: {len(semantic_correct)}/{n} → {semantic_correct}")
    print(f"    ❌ wrong:   {len(semantic_wrong)}/{n} → {semantic_wrong}")
    if errors:
        print(f"  💥 translation/transform errors: {len(errors)} → {errors}")
    print("=" * 72)

if __name__ == "__main__":
    main()