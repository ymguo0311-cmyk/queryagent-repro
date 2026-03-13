"""
pyql_to_cypher.py
─────────────────
Translates a PyQL operation list into a single Cypher query
for execution against a Neo4j Freebase backend.

PyQL operations (7 total):
    add_fact(s, relation, o)         → MATCH pattern
    add_type_constrain(type, var)    → label filter on var
    add_filter(var, op, value)       → WHERE clause
    add_count(var, count_var)        → COUNT(DISTINCT var)
    add_max(var)                     → ORDER BY var DESC LIMIT 1
    add_min(var)                     → ORDER BY var ASC LIMIT 1
    set_answer(var)                  → RETURN var

Freebase relation naming convention in Neo4j:
    Relation name: "people.person.date_of_birth"
    Neo4j rel type: `people__person__date_of_birth`   (dots → double underscores)

Entity node label: :Entity  with property mid = "m.0fjvv"
Relation node:     not used here (only used for vector ranking)

Usage:
    from pyql_to_cypher import PyQLToCypher
    translator = PyQLToCypher()
    cypher = translator.translate([
        "add_fact(m.0fjvv, astronomy.celestial_object_category.objects, ?star)",
        "add_fact(?star, astronomy.star.temperature_k, ?temperature)",
        "add_min(?temperature)",
        "set_answer(?star)"
    ])
    print(cypher)
"""

import re
from typing import List, Optional


# ── helpers ───────────────────────────────────────────────────────────────────

def rel_to_neo4j_type(relation: str) -> str:
    """
    Convert Freebase relation name to Neo4j relationship type.
    e.g. "people.person.date_of_birth" → "people__person__date_of_birth"
    Neo4j rel types cannot contain dots, so we use double underscores.
    """
    return relation.replace(".", "__")


def is_entity(token: str) -> bool:
    """Check if token is a Freebase entity MID (e.g. m.0fjvv, g.125abc)."""
    return bool(re.match(r'^[mg]\.[0-9a-zA-Z_]+$', token))


def is_variable(token: str) -> bool:
    """Check if token is a PyQL variable (starts with ?)."""
    return token.startswith("?")


def var_name(token: str) -> str:
    """Strip leading ? from variable name for use in Cypher."""
    return token.lstrip("?")


def mid_to_cypher_id(mid: str) -> str:
    """
    Convert Freebase MID to a safe Cypher identifier.
    e.g. "m.0fjvv" → "m_0fjvv"
    """
    return mid.replace(".", "_")


# ── parser ────────────────────────────────────────────────────────────────────

def parse_op(op_str: str):
    """
    Parse a single PyQL operation string into (op_name, args).

    Examples:
        "add_fact(m.0fjvv, astronomy.star.temperature_k, ?temperature)"
        → ("add_fact", ["m.0fjvv", "astronomy.star.temperature_k", "?temperature"])

        "add_type_constrain(astronomy.star, ?star)"
        → ("add_type_constrain", ["astronomy.star", "?star"])

        "add_min(?temperature)"
        → ("add_min", ["?temperature"])
    """
    op_str = op_str.strip().rstrip(")")
    paren_idx = op_str.index("(")
    op_name = op_str[:paren_idx].strip()
    args_str = op_str[paren_idx + 1:].strip()
    args = [a.strip() for a in args_str.split(",")]
    return op_name, args


# ── main translator class ─────────────────────────────────────────────────────

class PyQLToCypher:
    """
    Stateful translator: accumulates MATCH/WHERE clauses from PyQL ops,
    then emits a single Cypher query string when translate() is called.
    """

    def translate(self, pyql_ops: List[str]) -> str:
        """
        Main entry point.

        Args:
            pyql_ops: list of PyQL operation strings (as stored in current_pyql)

        Returns:
            A Cypher query string ready to run against Neo4j.
        """
        self._reset()

        for op_str in pyql_ops:
            try:
                op_name, args = parse_op(op_str)
            except Exception as e:
                raise ValueError(f"Cannot parse PyQL op: '{op_str}' — {e}")

            if op_name == "add_fact":
                self._handle_add_fact(args)
            elif op_name == "add_type_constrain":
                self._handle_add_type_constrain(args)
            elif op_name == "add_filter":
                self._handle_add_filter(args)
            elif op_name == "add_count":
                self._handle_add_count(args)
            elif op_name == "add_max":
                self._handle_add_max(args)
            elif op_name == "add_min":
                self._handle_add_min(args)
            elif op_name == "set_answer":
                self._handle_set_answer(args)
            else:
                raise ValueError(f"Unknown PyQL operation: {op_name}")

        return self._build_cypher()

    # ── internal state ────────────────────────────────────────────────────────

    def _reset(self):
        self.match_clauses: List[str] = []   # MATCH ... lines
        self.where_clauses: List[str] = []   # WHERE ... conditions
        self.return_var: Optional[str] = None
        self.count_var: Optional[str] = None  # if add_count used
        self.count_target: Optional[str] = None
        self.order_by: Optional[str] = None   # "ASC" or "DESC"
        self.order_var: Optional[str] = None
        self.limit_one: bool = False
        self.entity_nodes: dict = {}          # mid → cypher_id, already declared

    # ── op handlers ──────────────────────────────────────────────────────────

    def _handle_add_fact(self, args: List[str]):
        """
        add_fact(subject, relation, object)

        Cases:
          add_fact(m.0fjvv, rel, ?var)       → forward: entity -[rel]-> ?var
          add_fact(?var, rel, m.0fjvv)       → backward match with WHERE
          add_fact(?var1, rel, ?var2)        → variable to variable
          add_fact(m.0fjvv, rel, m.0abc)     → entity to entity (rare)
        """
        s, rel, o = args[0], args[1], args[2]
        rel_type = rel_to_neo4j_type(rel)

        s_cypher = self._node_ref(s)
        o_cypher = self._node_ref(o)

        self.match_clauses.append(
            f"MATCH ({s_cypher})-[:`{rel_type}`]->({o_cypher})"
        )

    def _handle_add_type_constrain(self, args: List[str]):
        """
        add_type_constrain(freebase.type, ?var)
        Adds a label constraint on ?var.
        In Neo4j we store the type as a property kg_type on the Entity node.
        e.g. add_type_constrain('astronomy.star', ?x)
        → WHERE x.kg_type = 'astronomy.star'  (or a MATCH with label)
        """
        fb_type, var = args[0], args[1]
        v = var_name(var)
        # Use WHERE property check (more flexible than label)
        self.where_clauses.append(f"{v}.kg_type = '{fb_type}'")

    def _handle_add_filter(self, args: List[str]):
        """
        add_filter(?var, operator, value)
        e.g. add_filter(?year, >=, 2000)
        → WHERE year >= 2000
        """
        var, op, value = args[0], args[1], args[2]
        v = var_name(var)
        self.where_clauses.append(f"{v} {op} {value}")

    def _handle_add_count(self, args: List[str]):
        """
        add_count(?var, ?count_var)
        Signals that the RETURN clause should use COUNT(DISTINCT var).
        """
        self.count_target = var_name(args[0])
        self.count_var = var_name(args[1])

    def _handle_add_max(self, args: List[str]):
        """
        add_max(?var)
        ORDER BY var DESC LIMIT 1
        """
        self.order_var = var_name(args[0])
        self.order_by = "DESC"
        self.limit_one = True

    def _handle_add_min(self, args: List[str]):
        """
        add_min(?var)
        ORDER BY var ASC LIMIT 1
        """
        self.order_var = var_name(args[0])
        self.order_by = "ASC"
        self.limit_one = True

    def _handle_set_answer(self, args: List[str]):
        """
        set_answer(?var)
        Sets the RETURN variable.
        """
        self.return_var = var_name(args[0])

    # ── node reference helper ────────────────────────────────────────────────

    def _node_ref(self, token: str) -> str:
        """
        Given a PyQL token (entity MID or ?variable), return a Cypher node pattern string.

        For entity MIDs (e.g. m.0fjvv):
            First use: `m_0fjvv:Entity {mid: 'm.0fjvv'}`
            Subsequent uses: `m_0fjvv`  (already declared)

        For variables (e.g. ?star):
            Returns `star`  (bare identifier, label added via WHERE if needed)
        """
        if is_variable(token):
            return var_name(token)

        if is_entity(token):
            cypher_id = mid_to_cypher_id(token)
            if cypher_id not in self.entity_nodes:
                self.entity_nodes[cypher_id] = token
                return f"{cypher_id}:Entity {{mid: '{token}'}}"
            else:
                return cypher_id

        # Fallback: treat as literal string node
        return f"n_{token.replace('.', '_')}"

    # ── cypher builder ────────────────────────────────────────────────────────

    def _build_cypher(self) -> str:
        """
        Assemble all collected clauses into a final Cypher query.
        """
        lines = []

        # MATCH clauses
        for m in self.match_clauses:
            lines.append(m)

        # WHERE clauses
        if self.where_clauses:
            lines.append("WHERE " + "\n  AND ".join(self.where_clauses))

        # RETURN clause
        if self.count_var and self.count_target:
            lines.append(
                f"RETURN COUNT(DISTINCT {self.count_target}) AS {self.count_var}"
            )
        elif self.return_var:
            lines.append(f"RETURN DISTINCT {self.return_var}")
        else:
            # fallback: return everything (shouldn't happen in practice)
            lines.append("RETURN *")

        # ORDER BY / LIMIT
        if self.order_by and self.order_var:
            lines.append(f"ORDER BY {self.order_var} {self.order_by}")
        if self.limit_one:
            lines.append("LIMIT 1")

        return "\n".join(lines)


# ── unit tests ────────────────────────────────────────────────────────────────

def run_tests():
    t = PyQLToCypher()

    print("=" * 60)
    print("TEST 1: red dwarf star with lowest temperature")
    print("=" * 60)
    ops1 = [
        "add_fact(m.0fjvv, astronomy.celestial_object_category.objects, ?star)",
        "add_fact(?star, astronomy.star.temperature_k, ?temperature)",
        "add_min(?temperature)",
        "set_answer(?star)"
    ]
    cypher1 = t.translate(ops1)
    print(cypher1)
    print()

    print("=" * 60)
    print("TEST 2: measurement system for watt per steradian")
    print("=" * 60)
    ops2 = [
        "add_fact(m.02sj5fc, measurement_unit.radiant_intensity_unit.measurement_system, ?measurement_system)",
        "set_answer(?measurement_system)"
    ]
    cypher2 = t.translate(ops2)
    print(cypher2)
    print()

    print("=" * 60)
    print("TEST 3: northern line terminuses count")
    print("=" * 60)
    ops3 = [
        "add_fact(m.0m_sb, metropolitan_transit.transit_line.terminuses, ?terminus)",
        "add_count(?terminus, ?count)",
        "set_answer(?count)"
    ]
    cypher3 = t.translate(ops3)
    print(cypher3)
    print()

    print("=" * 60)
    print("TEST 4: with type constraint")
    print("=" * 60)
    ops4 = [
        "add_type_constrain(astronomy.star, ?x)",
        "add_fact(?x, astronomy.celestial_object.category, m.0fjvv)",
        "add_fact(?x, astronomy.star.temperature_k, ?temp)",
        "add_min(?temp)",
        "set_answer(?x)"
    ]
    cypher4 = t.translate(ops4)
    print(cypher4)
    print()

    print("=" * 60)
    print("TEST 5: musical game requiring computer keyboard")
    print("=" * 60)
    ops5 = [
        "add_fact(?game, cvg.musical_game.input_method, m.01m2v)",
        "set_answer(?game)"
    ]
    cypher5 = t.translate(ops5)
    print(cypher5)
    print()

    print("=" * 60)
    print("TEST 6: galaxy code for barred spiral galaxy (max)")
    print("=" * 60)
    ops6 = [
        "add_fact(m.03q3pn, astronomy.galactic_shape.galaxies_of_this_shape, ?galaxy_code)",
        "set_answer(?galaxy_code)"
    ]
    cypher6 = t.translate(ops6)
    print(cypher6)
    print()

    print("=" * 60)
    print("TEST 7: add_filter example")
    print("=" * 60)
    ops7 = [
        "add_type_constrain(people.person, ?person)",
        "add_fact(?person, people.person.date_of_birth, ?dob)",
        "add_filter(?dob, >=, '2000-01-01')",
        "set_answer(?person)"
    ]
    cypher7 = t.translate(ops7)
    print(cypher7)
    print()


if __name__ == "__main__":
    run_tests()