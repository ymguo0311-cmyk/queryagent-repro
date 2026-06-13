"""
test_pyql_neo4j.py
──────────────────
Test PyQL → Cypher → Neo4j pipeline using 10 real GrailQA questions.
Run from ag_src directory:
    python3 agent_utils/test_pyql_neo4j.py
"""
import sys
sys.path.insert(0, '.')

from grail_src.pyql_to_cypher import PyQLToCypher
from agent_utils.config import NEO4J_DRIVER

test_cases = [
    {
        "idx": 1,
        "question": "which red dwarf stars star has the lowest temperature?",
        "pyql": [
            "add_fact(m.0fjvv, astronomy.celestial_object_category.objects, ?star)",
            "add_fact(?star, astronomy.star.temperature_k, ?temperature)",
            "add_min(?temperature)",
            "set_answer(?star)"
        ]
    },
    {
        "idx": 2,
        "question": "in which measurement system, watt per steradian is the radiant intensity unit?",
        "pyql": [
            "add_fact(m.02sj5fc, measurement_unit.radiant_intensity_unit.measurement_system, ?measurement_system)",
            "set_answer(?measurement_system)"
        ]
    },
    {
        "idx": 3,
        "question": "the permittivity units of farad per metre is part of what measurement system?",
        "pyql": [
            "add_fact(m.02sj567, measurement_unit.permittivity_unit.measurement_system, ?measurement_system)",
            "set_answer(?measurement_system)"
        ]
    },
    {
        "idx": 4,
        "question": "using amoxicillin for the treatment... which medical trial has the least number of expect total enrollment?",
        "pyql": [
            "add_fact(m.04d1kq9, medicine.medical_trial.treatment_being_tested, ?trial)",
            "add_fact(?trial, medicine.medical_treatment.trials, ?enrollment)",
            "set_answer(?trial)"
        ]
    },
    {
        "idx": 5,
        "question": "the container for digital negative shares the same genre of which file format?",
        "pyql": [
            "add_fact(m.03_2yh, computer.file_format.genre, ?genre)",
            "add_fact(?genre, computer.file_format_genre.file_formats, ?file_format)",
            "set_answer(?file_format)"
        ]
    },
    {
        "idx": 6,
        "question": "for what musical game do you need to have a computer keyboard?",
        "pyql": [
            "add_fact(m.01m2v, computer.computer_peripheral.supporting_games, ?game)",
            "add_fact(?game, cvg.computer_videogame.cvg_genre, ?genre)",
            "add_fact(?genre, media_common.media_genre.parent_genre, ?parent_genre)",
            "set_answer(?game)"
        ]
    },
    {
        "idx": 7,
        "question": "what is the software with genres editor and word processor?",
        "pyql": [
            "add_fact(m.082vy, computer.software_genre.software_in_genre, ?software)",
            "set_answer(?software)"
        ]
    },
    {
        "idx": 8,
        "question": "barred spiral galaxy is the shape of which galaxy code?",
        "pyql": [
            "add_fact(m.03q3pn, astronomy.galactic_shape.galaxies_of_this_shape, ?galaxy_code)",
            "set_answer(?galaxy_code)"
        ]
    },
    {
        "idx": 10,
        "question": "oersted is the magnetic field strength unit in what measurement system?",
        "pyql": [
            "add_fact(m.0fksj, measurement_unit.magnetic_field_strength_unit.measurement_system, ?measurement_system)",
            "set_answer(?measurement_system)"
        ]
    },
    {
        "idx": 11,
        "question": "which fictional character's species is virizion?",
        "pyql": [
            "add_fact(m.010g06mg, fictional_universe.character_species.characters_of_this_species, ?character)",
            "set_answer(?character)"
        ]
    }
]

translator = PyQLToCypher()
passed = 0
failed = 0

for tc in test_cases:
    print(f"\n{'='*60}")
    print(f"[{tc['idx']}] {tc['question']}")
    print(f"{'='*60}")

    try:
        cypher = translator.translate(tc['pyql'])
        print(f"Cypher:\n{cypher}\n")

        with NEO4J_DRIVER.session() as session:
            result = session.run(cypher)
            rows = list(result)

        if rows:
            keys = list(rows[0].keys())
            answers = []
            for row in rows[:5]:  # show max 5 results
                val = row[keys[0]]
                if hasattr(val, 'get'):
                    answers.append(val.get('mid', str(val)))
                else:
                    answers.append(str(val))
            print(f"✅ Results ({len(rows)} total): {answers}")
            passed += 1
        else:
            print(f"⚠️  No results returned")
            failed += 1

    except Exception as e:
        print(f"❌ Error: {e}")
        failed += 1

print(f"\n{'='*60}")
print(f"Summary: {passed} passed, {failed} failed out of {len(test_cases)} tests")
print(f"{'='*60}")
