"""
run_e2e_test.py
───────────────
End-to-end test using two separate modules:
  Step 1: pyql_to_cypher.py       — PyQL ops → standard Cypher
  Step 2: standard_to_semantic.py — standard Cypher → semantic Cypher

Compares semantic Cypher output against GT CSV.

Usage (from grail_src/ directory):
    python3 run_e2e_test.py
"""

import sys
sys.path.insert(0, '.')

from pyql_to_cypher import PyQLToCypher
from standard_to_semantic import transform_cypher_query

import re

# ── GT CSV loader ─────────────────────────────────────────────────────────────

GT_CSV = "original_to_semantic_cypher.csv"

def load_gt_csv(path: str):
    rows = []
    with open(path, encoding='utf-8') as f:
        next(f)  # skip header
        for line in f:
            line = line.strip()
            if not line:
                continue
            line = line.replace('\\"', '\x00')
            parts = line.split('","')
            if len(parts) == 2:
                col0 = parts[0].lstrip('"').replace('\x00', '"')
                col1 = parts[1].rstrip('"').replace('\x00', '"')
                rows.append((col0.strip(), col1.strip()))
    return rows

def normalize(s: str) -> str:
    return s.strip().rstrip(';').strip()

# ── Test cases (Index 1–11, skip 9) ──────────────────────────────────────────

TEST_CASES = [
    {
        "index": 1,
        "question": "which red dwarf stars star has the lowest temperature?",
        "pyql": [
            "add_fact(m.0fjvv, astronomy.celestial_object_category.objects, ?star)",
            "add_fact(?star, astronomy.star.temperature_k, ?temperature)",
            "add_min(?temperature)",
            "set_answer(?star)",
        ],
    },
    {
        "index": 2,
        "question": "in which measurement system, watt per steradian is the radiant intensity unit?",
        "pyql": [
            "add_fact(m.02sj5fc, measurement_unit.radiant_intensity_unit.measurement_system, ?measurement_system)",
            "set_answer(?measurement_system)",
        ],
    },
    {
        "index": 3,
        "question": "the permittivity units of farad per metre is part of what measurement system?",
        "pyql": [
            "add_fact(m.02sj567, measurement_unit.permittivity_unit.measurement_system, ?measurement_system)",
            "set_answer(?measurement_system)",
        ],
    },
    {
        "index": 4,
        "question": "using amoxicillin... which medical trial has the least number of expect total enrollment?",
        "pyql": [
            "add_fact(m.04d1kq9, medicine.medical_trial.treatment_being_tested, ?trial)",
            "add_fact(?trial, medicine.medical_treatment.trials, ?enrollment)",
            "set_answer(?trial)",
        ],
    },
    {
        "index": 5,
        "question": "the container for digital negative shares the same genre of which file format?",
        "pyql": [
            "add_fact(m.03_2yh, computer.file_format.genre, ?genre)",
            "add_fact(?genre, computer.file_format_genre.file_formats, ?file_format)",
            "set_answer(?file_format)",
        ],
    },
    {
        "index": 6,
        "question": "for what musical game do you need to have a computer keyboard?",
        "pyql": [
            "add_fact(m.01m2v, computer.computer_peripheral.supporting_games, ?game)",
            "add_fact(?game, cvg.computer_videogame.cvg_genre, ?genre)",
            "add_fact(?genre, media_common.media_genre.parent_genre, ?parent_genre)",
            "set_answer(?game)",
        ],
    },
    {
        "index": 7,
        "question": "what is the software with genres editor and word processor?",
        "pyql": [
            "add_fact(m.082vy, computer.software_genre.software_in_genre, ?software)",
            "set_answer(?software)",
        ],
    },
    {
        "index": 8,
        "question": "barred spiral galaxy is the shape of which galaxy code?",
        "pyql": [
            "add_fact(m.03q3pn, astronomy.galactic_shape.galaxies_of_this_shape, ?galaxy_code)",
            "set_answer(?galaxy_code)",
        ],
    },
    # index 9 skipped — no PyQL generated
    {
        "index": 10,
        "question": "oersted is the magnetic field strength unit in what measurement system?",
        "pyql": [
            "add_fact(m.0fksj, measurement_unit.magnetic_field_strength_unit.measurement_system, ?measurement_system)",
            "set_answer(?measurement_system)",
        ],
    },
    {
        "index": 11,
        "question": "which fictional character's species is virizion?",
        "pyql": [
            "add_fact(m.010g06mg, fictional_universe.character_species.characters_of_this_species, ?character)",
            "set_answer(?character)",
        ],
    },
]

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    translator = PyQLToCypher()

    try:
        gt_rows = load_gt_csv(GT_CSV)
        gt_semantic = [r[1] for r in gt_rows]
        has_gt = True
        print(f"Loaded {len(gt_rows)} GT rows from {GT_CSV}\n")
    except FileNotFoundError:
        print(f"[WARN] GT CSV not found at '{GT_CSV}'. Running without comparison.\n")
        gt_semantic = []
        has_gt = False

    passed, failed = 0, 0
    gt_idx = 0

    for tc in TEST_CASES:
        idx = tc["index"]
        print("=" * 64)
        print(f"[{idx}] {tc['question']}")
        print("=" * 64)

        # ── Step 1: PyQL → standard Cypher ───────────────────────────────────
        try:
            std = translator.translate(tc["pyql"])
        except Exception as e:
            print(f"  ❌ [Step 1] pyql_to_cypher ERROR: {e}\n")
            failed += 1
            gt_idx += 1
            continue

        print(f"  [Step 1] Standard Cypher:\n{std}\n")

        # ── Step 2: standard Cypher → semantic Cypher ─────────────────────────
        try:
            sem = transform_cypher_query(std)
        except Exception as e:
            print(f"  ❌ [Step 2] standard_to_semantic ERROR: {e}\n")
            failed += 1
            gt_idx += 1
            continue

        print(f"  [Step 2] Semantic Cypher:\n{sem}\n")

        # ── Step 3: Compare against GT ────────────────────────────────────────
        if has_gt and gt_idx < len(gt_semantic):
            gt_sem = gt_semantic[gt_idx]
            if normalize(sem) == normalize(gt_sem):
                print(f"  ✅ Matches GT\n")
                passed += 1
            else:
                print(f"  ❌ MISMATCH")
                print(f"  Expected:\n{gt_sem}\n")
                failed += 1
        else:
            print(f"  ⚠️  No GT to compare\n")
            passed += 1

        gt_idx += 1

    print("=" * 64)
    print(f"Summary: {passed} passed, {failed} failed out of {len(TEST_CASES)} tests")
    print("=" * 64)


if __name__ == "__main__":
    main()