"""
standard_to_semantic.py
───────────────────────
Converts standard Cypher to semantic Cypher using semantic.matchSubgraph UDP.

Transformations:
  1. ORDER BY / LIMIT separated to outer clause
  2. Multiple MATCH clauses merged into single comma-separated MATCH
  3. Edge labels [:`REL`] → [{~description: 'REL'}]
  4. ORDER BY variables prefixed with row.
  5. RETURN variables prefixed with row.

Usage:
    from standard_to_semantic import transform_cypher_query
"""

import re


def transform_cypher_query(q: str) -> str:
    """
    Convert a standard Cypher query to semantic Cypher.

    Args:
        q: standard Cypher query string

    Returns:
        semantic Cypher string using semantic.matchSubgraph UDP
    """
    q = q.strip()
    if q.endswith(';'):
        q = q[:-1]

    # Step 1: Split off tail clauses (ORDER BY, LIMIT, SKIP) — keep outside CALL
    tail_match = re.search(r'(?i)\b(ORDER\s+BY|LIMIT|SKIP)\b', q)
    if tail_match:
        tail_clause = q[tail_match.start():]
        q = q[:tail_match.start()].strip()
    else:
        tail_clause = ""
    
  
    # Step 2: Split on last RETURN
    parts = re.split(r'(?i)\bRETURN\b', q)
    if len(parts) < 2:
        raise ValueError("No RETURN clause found in the query.")
    match_body = "RETURN".join(parts[:-1]).strip()
    return_clause = parts[-1].strip()

    # Step 3: Merge multiple MATCH clauses into one comma-separated MATCH
    segments = re.split(r'(?i)\bMATCH\b', match_body)
    segments = [s.strip() for s in segments if s.strip()]
    merged_match = "MATCH " + ", ".join(segments)

    # Step 4: Convert edge labels [:`REL`] → [{~description: 'REL'}]
    merged_match = re.sub(
        r'\[:`([^`]+)`\]',
        lambda m: "[{~description: '" + m.group(1) + "'}]",
        merged_match
    )

    # Step 5: Add row. prefix to ORDER BY expressions (format: var.prop)
    tail_clause = re.sub(
        r'(?i)(ORDER\s+BY\s+)(\w+\.\w+)',
        lambda m: m.group(1) + "row." + m.group(2),
        tail_clause
    )

    # Step 6: Handle DISTINCT in RETURN + special case
    distinct_prefix = ""
    distinct_match = re.match(r'(?i)^DISTINCT\s+', return_clause)
    if distinct_match:
        distinct_prefix = "DISTINCT "
        return_clause = return_clause[len(distinct_match.group(0)):].strip()
    
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

    # Step 7: Safe-split return items (respects parens)
    items, current, paren_level = [], [], 0
    for char in return_clause:
        if char == '(':
            paren_level += 1
        elif char == ')':
            paren_level -= 1
        elif char == ',' and paren_level == 0:
            items.append("".join(current).strip())
            current = []
            continue
        current.append(char)
    if current:
        items.append("".join(current).strip())

    # Step 8: Add row. prefix to each return item
    new_items = [f"row.{item}" for item in items if item]
    new_return = distinct_prefix + ", ".join(new_items)

    # Step 9: Escape inner query double quotes
    inner_cypher = merged_match.replace('"', '\\"')

    # Step 10: Assemble
    result = f'CALL semantic.matchSubgraph("{inner_cypher} RETURN *", true) YIELD row RETURN {new_return}'
    if tail_clause:
        result += f" {tail_clause}"
    result += ";"

    return result