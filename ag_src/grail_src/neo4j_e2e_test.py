"""
neo4j_e2e_test.py
─────────────────
End-to-end test: PyQL → standard Cypher → semantic Cypher → Neo4j execution

Run this on the lab server (cpu11) where Neo4j is accessible.

Requirements:
    pip install neo4j

Usage:
    python3 neo4j_e2e_test.py

Adjust NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD below if needed.
"""

import re
from neo4j import GraphDatabase
from pyql_to_cypher import PyQLToCypher

# ── Neo4j connection config ───────────────────────────────────────────────────
NEO4J_URI      = "neo4j://localhost:7690"
NEO4J_USER     = "neo4j"
NEO4J_PASSWORD = "12345678"

# ── standard → semantic converter ────────────────────────────────────────────

def transform_cypher_query(q: str) -> str:
    """
    Convert standard Cypher to semantic Cypher using semantic.matchSubgraph UDP.
    Replace this function with Michelle's version when available.
    """
    q = q.strip()
    if q.endswith(';'):
        q = q[:-1]

    # Split off tail clauses (ORDER BY, LIMIT, SKIP)
    tail_match = re.search(r'(?i)\b(ORDER\s+BY|LIMIT|SKIP)\b', q)
    if tail_match:
        tail_clause = q[tail_match.start():]
        q = q[:tail_match.start()].strip()
    else:
        tail_clause = ""

    # Split on last RETURN
    parts = re.split(r'(?i)\bRETURN\b', q)
    if len(parts) < 2:
        raise ValueError("No RETURN clause found.")
    match_body = "RETURN".join(parts[:-1]).strip()
    return_clause = parts[-1].strip()

    # Merge multiple MATCH clauses into one comma-separated MATCH
    segments = re.split(r'(?i)\bMATCH\b', match_body)
    segments = [s.strip() for s in segments if s.strip()]
    merged_match = "MATCH " + ", ".join(segments)

    # Convert edge labels [:`REL`] → [{~description: 'REL'}]
    merged_match = re.sub(
        r'\[:`([^`]+)`\]',
        lambda m: "[{~description: '" + m.group(1) + "'}]",
        merged_match
    )

    # Add row. prefix to ORDER BY expressions
    tail_clause = re.sub(
        r'(?i)(ORDER\s+BY\s+)(\w+\.\w+)',
        lambda m: m.group(1) + "row." + m.group(2),
        tail_clause
    )

    # Handle DISTINCT
    distinct_prefix = ""
    dm = re.match(r'(?i)^DISTINCT\s+', return_clause)
    if dm:
        distinct_prefix = "DISTINCT "
        return_clause = return_clause[len(dm.group(0)):].strip()

    # Handle COUNT(DISTINCT ...) specially — don't prepend row. to the whole thing
    count_match = re.match(r'(?i)^(COUNT\s*\(\s*DISTINCT\s+(\w+)\s*\))\s+AS\s+(\w+)$', return_clause)
    if count_match:
        inner_var = count_match.group(2)
        alias = count_match.group(3)
        inner_cypher = merged_match.replace('"', '\\"')
        result = f'CALL semantic.matchSubgraph("{inner_cypher} RETURN *", true) YIELD row RETURN COUNT(DISTINCT row.{inner_var}) AS {alias}'
        if tail_clause:
            result += f" {tail_clause}"
        result += ";"
        return result

    # Safe-split return items
    items, current, paren_level = [], [], 0
    for char in return_clause:
        if char == '(':   paren_level += 1
        elif char == ')': paren_level -= 1
        elif char == ',' and paren_level == 0:
            items.append("".join(current).strip())
            current = []
            continue
        current.append(char)
    if current:
        items.append("".join(current).strip())

    new_items = [f"row.{item}" for item in items if item]
    new_return = distinct_prefix + ", ".join(new_items)

    inner_cypher = merged_match.replace('"', '\\"')
    result = f'CALL semantic.matchSubgraph("{inner_cypher} RETURN *", true) YIELD row RETURN {new_return}'
    if tail_clause:
        result += f" {tail_clause}"
    result += ";"

    return result


# ── Test cases ────────────────────────────────────────────────────────────────

TEST_CASES = [
    {
        "index": 12,
        "question": "What was the name of the rocket with the smallest diameter?",
        "gt_answer": ["m.0g9xpty"],
        "pyql": [
            "add_type_constrain(spaceflight.rocket, ?x0)",
            "add_fact(?x0, spaceflight.rocket.diameter_meters, ?x1)",
            "add_min(?x1)",
            "set_answer(?x0)",
        ],
    },
    {
        "index": 13,
        "question": "The developer of Battle Forge also developed what video game version?",
        "gt_answer": ["m.05nspt0", "m.0kyq3h2", "m.0kyq5k0", "m.0kz27qj"],
        "pyql": [
            "add_type_constrain(cvg.game_version, ?x0)",
            "add_fact(m.04f56nl, cvg.computer_videogame.developer, ?x1)",
            "add_fact(?x0, cvg.game_version.developer, ?x1)",
            "set_answer(?x0)",
        ],
    },
    {
        "index": 14,
        "question": "Virgins is the subject of what written work?",
        "gt_answer": ["m.06hns8m"],
        "pyql": [
            "add_type_constrain(book.written_work, ?x0)",
            "add_fact(?x0, book.written_work.subjects, m.0mcgd)",
            "set_answer(?x0)",
        ],
    },
    {
        "index": 15,
        "question": "What channel access method has a child method of a channel access method with a parent method of packet mode multiple access?",
        "gt_answer": ["m.012vrj", "m.026lnjl", "m.0h0fwx"],
        "pyql": [
            "add_type_constrain(engineering.channel_access_method, ?x0)",
            "add_fact(?x1, engineering.channel_access_method.parent_method, m.05y8dl3)",
            "add_fact(?x1, engineering.channel_access_method.child_method, ?x0)",
            "set_answer(?x0)",
        ],
    },
    {
        "index": 16,
        "question": "What was the unit of heat capacity that is used in the international system of units system?",
        "gt_answer": ["m.02sj4xx"],
        "pyql": [
            "add_type_constrain(measurement_unit.heat_capacity_unit, ?x0)",
            "add_fact(?x0, measurement_unit.heat_capacity_unit.measurement_system, m.0c13h)",
            "set_answer(?x0)",
        ],
    },
    {
        "index": 17,
        "question": "What was Gimme Shelter a video of?",
        "gt_answer": ["m.01t0by"],
        "pyql": [
            "add_type_constrain(music.concert, ?x0)",
            "add_fact(?x0, music.concert.concert_video, m.091w4_)",
            "set_answer(?x0)",
        ],
    },
    {
        "index": 18,
        "question": "For deracoxib what is the number of contraindications?",
        "gt_answer": ["1"],
        "pyql": [
            "add_type_constrain(medicine.contraindication, ?x1)",
            "add_fact(?x1, medicine.contraindication.contraindication_for, m.026s109)",
            "add_count(?x1, ?x0)",
        ],
    },
    {
        "index": 19,
        "question": "Name the video game that has subjects of hammer throwing in it.",
        "gt_answer": ["m.02q8x4x"],
        "pyql": [
            "add_type_constrain(cvg.computer_videogame, ?x0)",
            "add_fact(?x0, cvg.computer_videogame.subjects, m.0byp9)",
            "set_answer(?x0)",
        ],
    },
    {
        "index": 20,
        "question": "8000.0 rate in bits per second is for what unit of data transmission rate?",
        "gt_answer": ["m.02wv8g3"],
        "pyql": [
            "add_type_constrain(measurement_unit.unit_of_data_transmission_rate, ?x0)",
            'add_fact(?x0, measurement_unit.unit_of_data_transmission_rate.rate_in_bits_per_second, "8000.0"^^xsd:float)',
            "set_answer(?x0)",
        ],
    },
    {
        "index": 21,
        "question": "San Jose Convention Center hosted which conference?",
        "gt_answer": ["m.06mrx_k"],
        "pyql": [
            "add_type_constrain(conferences.conference, ?x0)",
            "add_fact(?x0, conferences.conference.venue, m.03nvnck)",
            "set_answer(?x0)",
        ],
    },
    {
        "index": 22,
        "question": "For cameras with camera iso capability high iso auto, what camera image stabilization type is used?",
        "gt_answer": ["m.022q2pk"],
        "pyql": [
            "add_type_constrain(digicams.image_stabilization_type, ?x0)",
            "add_fact(m.02nqg65, digicams.camera_iso.cameras, ?x1)",
            "add_fact(?x0, digicams.image_stabilization_type.digital_camera, ?x1)",
            "set_answer(?x0)",
        ],
    },
    {
        "index": 23,
        "question": "Which locomotives are members of the british rail class 37 class?",
        "gt_answer": ["m.0h0fmy"],
        "pyql": [
            "add_type_constrain(rail.locomotive, ?x0)",
            "add_fact(?x0, rail.locomotive.locomotive_class, m.051nwx)",
            "set_answer(?x0)",
        ],
    },
    {
        "index": 24,
        "question": "Christopher Blackett owns what kind of locomotive?",
        "gt_answer": ["m.02l_nw"],
        "pyql": [
            "add_type_constrain(rail.locomotive, ?x0)",
            "add_fact(?x1, rail.locomotive_ownership.owner, m.05blpgl)",
            "add_fact(?x1, rail.locomotive_ownership.locomotive, ?x0)",
            "set_answer(?x0)",
        ],
    },
    {
        "index": 25,
        "question": "What ship designer designed a ship that is designed by Peter Pett?",
        "gt_answer": ["m.0341fb"],
        "pyql": [
            "add_type_constrain(boats.ship_designer, ?x0)",
            "add_fact(?x1, boats.ship.designer, m.033jrn)",
            "add_fact(?x0, boats.ship_designer.boats_designed, ?x1)",
            "set_answer(?x0)",
        ],
    },
    {
        "index": 26,
        "question": "Of time zones, which has a dst offset from utc of more than 11.0?",
        "gt_answer": ["m.05br69j", "m.09zykc", "m.0y49bcg", "m.0y4d_vm"],
        "pyql": [
            "add_type_constrain(time.time_zone, ?x0)",
            "add_fact(?x0, time.time_zone.dst_offset_from_utc, ?x1)",
            'add_filter(?x1, >, 11.0)',
            "set_answer(?x0)",
        ],
    },
    {
        "index": 27,
        "question": "What measurement systems is watt time unit of?",
        "gt_answer": ["m.0c13h"],
        "pyql": [
            "add_type_constrain(measurement_unit.measurement_system, ?x0)",
            "add_fact(?x0, measurement_unit.measurement_system.power_units, m.09hl5)",
            "set_answer(?x0)",
        ],
    },
    {
        "index": 28,
        "question": "Please explain the unit of viscosity in the international system of units?",
        "gt_answer": ["m.02sj4qd"],
        "pyql": [
            "add_type_constrain(measurement_unit.viscosity_unit, ?x0)",
            "add_fact(?x0, measurement_unit.viscosity_unit.measurement_system, m.0c13h)",
            "set_answer(?x0)",
        ],
    },
    {
        "index": 29,
        "question": "What is the organism classification for the fossil Bodo cranium?",
        "gt_answer": ["m.02gj4w"],
        "pyql": [
            "add_type_constrain(biology.organism_classification, ?x0)",
            "add_fact(?x0, biology.organism_classification.fossil_specimens, m.0n8_wf9)",
            "set_answer(?x0)",
        ],
    },
    {
        "index": 30,
        "question": "Pulseman can be classified under what genre of video games?",
        "gt_answer": ["m.025zzc", "m.07yqn"],
        "pyql": [
            "add_type_constrain(cvg.cvg_genre, ?x0)",
            "add_fact(?x0, cvg.cvg_genre.games, m.083t1_)",
            "set_answer(?x0)",
        ],
    },
    {
        "index": 31,
        "question": "International Games System published what video game?",
        "gt_answer": ["m.02qfrm1", "m.04j9y7m", "m.0dc0mm"],
        "pyql": [
            "add_type_constrain(cvg.computer_videogame, ?x0)",
            "add_fact(?x0, cvg.computer_videogame.publisher, m.0ds98f)",
            "set_answer(?x0)",
        ],
    },
]


# ── result comparison ─────────────────────────────────────────────────────────

def compare(pred, gt):
    """Check if predicted results match ground truth (order-insensitive)."""
    pred_set = set(str(x).strip() for x in pred)
    gt_set   = set(str(x).strip() for x in gt)
    return pred_set == gt_set


# ── Neo4j runner ──────────────────────────────────────────────────────────────

def run_query(driver, cypher: str):
    """Execute a Cypher query and return flat list of result values."""
    with driver.session() as session:
        result = session.run(cypher)
        rows = result.data()
        if not rows:
            return []
        # Flatten: collect all values from all rows
        values = []
        for row in rows:
            for v in row.values():
                if hasattr(v, 'get'):
                    # Neo4j node — try to get mid, else str
                    values.append(v.get('mid', str(v)))
                else:
                    values.append(str(v))
        return values


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    translator = PyQLToCypher()

    print(f"Connecting to Neo4j at {NEO4J_URI} ...")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        driver.verify_connectivity()
        print("Connected.\n")
    except Exception as e:
        print(f"❌ Cannot connect to Neo4j: {e}")
        return

    passed, failed, errors = [], [], []

    for tc in TEST_CASES:
        idx = tc["index"]
        print("=" * 64)
        print(f"[{idx}] {tc['question']}")
        print(f"  GT: {tc['gt_answer']}")
        print("=" * 64)

        # Step 1: PyQL → standard Cypher
        try:
            std = translator.translate(tc["pyql"])
        except Exception as e:
            print(f"  ❌ Translation error: {e}\n")
            errors.append(idx)
            continue

        print(f"  Standard Cypher:\n{std}\n")

        # Step 2: standard → semantic Cypher
        try:
            sem = transform_cypher_query(std)
        except Exception as e:
            print(f"  ❌ transform error: {e}\n")
            errors.append(idx)
            continue

        print(f"  Semantic Cypher:\n{sem}\n")

        # Step 3: Execute against Neo4j
        try:
            pred = run_query(driver, sem)
        except Exception as e:
            print(f"  ❌ Neo4j error: {e}\n")
            errors.append(idx)
            continue

        print(f"  Result: {pred}")

        if compare(pred, tc["gt_answer"]):
            print(f"  ✅ CORRECT\n")
            passed.append(idx)
        else:
            print(f"  ❌ WRONG (expected {tc['gt_answer']})\n")
            failed.append(idx)

    driver.close()

    print("=" * 64)
    print(f"✅ Passed:  {len(passed)}/{len(TEST_CASES)}  → {passed}")
    print(f"❌ Failed:  {len(failed)}  → {failed}")
    print(f"💥 Errors:  {len(errors)}  → {errors}")
    print("=" * 64)


if __name__ == "__main__":
    main()
