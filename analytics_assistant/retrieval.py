from collections import deque
import json
import re

DESCRIPTIONS = {
 "CUSTOMER": "customers market segments account balances customer nations",
 "ORDERS": "orders dates priorities order status total prices customer orders",
 "LINEITEM": "sales discounted revenue quantities shipments delivery receipt commit discounts taxes",
 "SUPPLIER": "suppliers account balances supplier nations",
 "PART": "parts products brands types sizes retail prices containers",
 "PARTSUPP": "supplier part supply cost available inventory stock value",
 "NATION": "nations countries names region membership",
 "REGION": "regions geography Africa America Asia Europe Middle East",
}
JOINS = [
 ("CUSTOMER", "C_CUSTKEY", "ORDERS", "O_CUSTKEY"),
 ("ORDERS", "O_ORDERKEY", "LINEITEM", "L_ORDERKEY"),
 ("LINEITEM", "L_PARTKEY", "PART", "P_PARTKEY"),
 ("LINEITEM", "L_SUPPKEY", "SUPPLIER", "S_SUPPKEY"),
 ("PARTSUPP", "PS_PARTKEY", "PART", "P_PARTKEY"),
 ("PARTSUPP", "PS_SUPPKEY", "SUPPLIER", "S_SUPPKEY"),
 ("CUSTOMER", "C_NATIONKEY", "NATION", "N_NATIONKEY"),
 ("SUPPLIER", "S_NATIONKEY", "NATION", "N_NATIONKEY"),
 ("NATION", "N_REGIONKEY", "REGION", "R_REGIONKEY"),
]


def tokens(text):
    words = re.findall(r"[a-z][a-z0-9]*", text.lower().replace("_", " "))
    return {word[:-1] if word.endswith("s") and len(word) > 3 else word for word in words}


def overlap(question, text):
    q, t = tokens(question), tokens(text)
    return len(q & t) / max(1, len(q | t))


class Retriever:
    def __init__(self, schema):
        self.schema = schema
        self.graph = {table: set() for table in schema}
        for left, _, right, _ in JOINS:
            self.graph[left].add(right); self.graph[right].add(left)

    def path(self, start, target):
        queue = deque([[start]])
        seen = {start}
        while queue:
            path = queue.popleft()
            if path[-1] == target:
                return path
            for table in sorted(self.graph[path[-1]]):
                if table not in seen:
                    seen.add(table); queue.append(path + [table])
        return []

    def retrieve(self, question):
        ranked = sorted(self.schema, key=lambda table: (-overlap(question, table + " " + DESCRIPTIONS[table] + " " + json.dumps(self.schema[table])), table))
        chosen = set(ranked[:3])
        geographic = {"AFRICA", "AMERICA", "ASIA", "EUROPE", "MIDDLE EAST"}
        if any(name in question.upper() for name in geographic):
            chosen.add("REGION")
        nations = {"ALGERIA", "ARGENTINA", "BRAZIL", "CANADA", "EGYPT", "ETHIOPIA", "FRANCE", "GERMANY", "INDIA", "INDONESIA", "IRAN", "IRAQ", "JAPAN", "JORDAN", "KENYA", "MOROCCO", "MOZAMBIQUE", "PERU", "CHINA", "ROMANIA", "SAUDI ARABIA", "VIETNAM", "RUSSIA", "UNITED KINGDOM", "UNITED STATES"}
        if any(name in question.upper() for name in nations):
            chosen.add("NATION")
        for table in sorted(chosen - {ranked[0]}):
            chosen.update(self.path(ranked[0], table))
        schemas = {table: self.schema[table] for table in sorted(chosen)}
        joins = [f"{a}.{ak} = {b}.{bk}" for a, ak, b, bk in JOINS if a in chosen and b in chosen]
        return {"schemas": schemas, "join_hints": joins}
