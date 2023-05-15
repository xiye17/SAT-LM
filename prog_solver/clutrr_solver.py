import sys
sys.path.append('.')

import os
import pickle
from collections import OrderedDict

from prog_solver.z3_utils import execute_z3_test


def read_proglm_rules(filename="prog_solver/external/clutrr_proglm_rules.pkl"):
    with open("prog_solver/external/clutrr_proglm_rules.pkl", 'rb') as f:
        return pickle.load(f)

PROGLM_RULES = read_proglm_rules()

def read_satlm_rules(filename="prog_solver/external/clutrr_satlm_rules.txt"):
    d = OrderedDict()
    with open(filename) as f:
        lines = f.readlines()
    for l in lines:
        lhs, rhs = l.split("->")
        a, b = lhs.split()
        targets = tuple(rhs.split())
        d[(a, b)] = targets
    return d


def clutrr_proglm_exec(output_code):
    relations = []
    for line in output_code.split('\n')[1:]:
        if not line.startswith('#') and not '@' in line:
            try:
                relation = line.split(' = ')[1]
            except IndexError as e:
                continue
            relations.append(relation)

    final_relation = ""

    for relation in reversed(relations):
        if not final_relation:
            final_relation = relation
        else:
            # transitive rules
            final_relation = PROGLM_RULES[(final_relation, relation)]
    return final_relation


def prepare_sound_transitive_constraints():
    for (k1, k2), v in PROGLM_RULES.items():
        if "self" in (k1, k2, v):
            continue
        print(f"{k1} {k2} -> {v}")


SAT_STATES = {}
def construct_sat_states():
    if SAT_STATES:
        return SAT_STATES
    relations = []

    for k, v in PROGLM_RULES.items():
        items = k + (v,)
        for item in items:
            if item not in relations:
                relations.append(item)

    relations = [r.replace("-", "_") for r in relations]
    # relations = relations + ["distant"]

    decl = []
    decl.append("RelationSort, ({}) = EnumSort('RelationSort', [{}])".format(
        ", ".join(relations),
        ", ".join([f"'{n}'" for n in relations])
    ))
    decl.append("relation_names = [{}]".format(", ".join([f"'{n}'" for n in relations])))
    decl.append("relations = [{}]".format(", ".join(relations)))

    decl.append("R = Function('R', PeopleSort, PeopleSort, RelationSort)")

    transitive_constraints = []
    transitive_constraints.append("cer_precond = []")
    transitive_constraints.append("x, y, z = Consts('x y z', PeopleSort)")
    # transitive_constraints.append(f"x, y = FreshConst(PeopleSort), FreshConst(PeopleSort)")
    transitive_constraints.append("cer_precond.append(ForAll([x, y], (x == y) == (R(x, y) == self)))")

    certified_trans_eules = read_satlm_rules()
    for (k1, k2), vs in certified_trans_eules.items():
        k1 = k1.replace("-", "_")
        k2 = k2.replace("-", "_")
        vs = (v.replace("-", "_") for v in vs)
        cons = f"ForAll([x, y, z], Implies(And(x != z, R(x, y) == {k1}, R(y, z) == {k2}), Or({', '.join([f'R(x, z) == {v}' for v in vs])})))"
        transitive_constraints.append(f"cer_precond.append({cons})")
    # unique relations
    unqique_relations = ["father", "mother", "husband", "wife"]
    for r in unqique_relations:
        transitive_constraints.append(f"cer_precond.append(ForAll([x, y, z], Implies(And(R(x, y) == {r}, R(x, z) == {r}), y == z)))")
    reflexive_relations = [
        (("father", "mother"), ("son", "daughter")),
        (("brother", "sister"), ("brother", "sister")),
        (("husband",), ("wife",)),]
    for a, b in reflexive_relations:
        transitive_constraints.append(f"cer_precond.append(ForAll([x, y], Or({', '.join([f'R(x, y) == {r}' for r in a])}) == Or({', '.join([f'R(y, x) == {r}' for r in b])})))")

    SAT_STATES['relations'] = relations
    SAT_STATES['decl'] = decl
    SAT_STATES['constraints'] = transitive_constraints
    return SAT_STATES

def parse_clutrr_sat_problem(code, prompting_style, return_code=False):
    assert prompting_style == "satlm"
    states = construct_sat_states()
    relations, relation_decl, constraints = (
        states['relations'], states['decl'], states['constraints']
    )

    lines = code.splitlines()
    lines = [l.strip() for l in lines]
    lines = [l for l in lines if not l.startswith('#')]

    assert lines[-1].startswith('return')
    cons_lines = lines[:-1]
    return_line = lines[-1]

    q0, q1 = return_line.split('return relation')[1][1:-1].split(', ')
    names = []
    rel_lines = []
    
    for l in cons_lines:
        lhs, rhs = l.split(' = ')
        assert lhs.startswith('relation')
        lhs = lhs.split('relation')[1][1:-1]
        n1, n2 = lhs.split(', ')
        r1, r2 = rhs[1:-1].split(', ')
        if n1 not in names:
            names.append(n1)
        if n2 not in names:
            names.append(n2)
        if r2 in relations:
            rel_lines.append(f"R({n1}, {n2}) == {r2}")
        if r1 in relations:
            rel_lines.append(f"R({n2}, {n1}) == {r1}")

    translated_lines = []
    translated_lines.append("PeopleSort, ({}) = EnumSort('PeopleSort', [{}])".format(
        ", ".join([f"{n}" for n in names]),
        ", ".join([f"'{n}'" for n in names])
    ))
    translated_lines.append(f"q0 = {q0}")
    translated_lines.append(f"q1 = {q1}")
    translated_lines.extend(relation_decl)
    for t in constraints:
        translated_lines.append(t)
    
    translated_lines.append("decl_conditions = []")
    for l in rel_lines:
        translated_lines.append(f"decl_conditions.append({l})")

    if not return_code:
        translated_lines.extend([
            "s = Solver()",
            "s.add(cer_precond)",
            "s.add(decl_conditions)",
            "s.check()",
            "answers = []",
            "while s.check() == sat:",
            "    a = s.model().eval(R(q1, q0))",
            "    answers.append(a)",
            "    s.add(Not(R(q1, q0) == a))",
            "if answers:",
            "    print(answers)",
            "else:",
            "    print('UNSAT')",
        ])
    else:
        # statis style
        translated_lines.extend([
            "s = Solver()",
            "s.add(decl_conditions)",
            "s.add(cer_precond)",
            "if s.check(sound_precond) == sat:",
            "    print(s.model().eval(R(q1, q0)))",
            "else:",
            "    print('UNSAT')",
            "    print(s.unsat_core())"
        ])

    translated_code = ["from z3 import *"] + translated_lines

    return translated_code

def clutrr_satlm_exec(code, prompting_style, return_code=False):
    translated_code = parse_clutrr_sat_problem(code, prompting_style, return_code=return_code)
    translated_code = "\n".join(translated_code)
    result = execute_z3_test(translated_code, timeout=5.0)
    result = (result[0], result[1].replace("_", "-"))
    if return_code:
        return translated_code, result
    status, answer = result

    if answer.startswith("[") and answer.endswith("]"):
        answer = answer[1:-1].split(", ")
        if len(answer) > 3:
            answer = "AMBIG"
        else:
            answer.sort(key=lambda x: len(x))
            answer = answer[0]
        return status, answer
    else:
        return result

def ex_test():
    code = """
	# [Michael] was thrilled his brother, [Peter], was able to make it to the party.
    relation(Michael, Peter) = (brother, brother)
    # [Carmelita] spent a great day shopping with her daughter, [Martha].
    relation(Carmelita, Martha) = (mother, daughter)
    # [Brandi] watched a golf tournament with her aunt [Elizabeth].
    relation(Brandi, Elizabeth) = (niece, aunt)
    # [Chuck] and [Elizabeth] got married in Hawaii.
    relation(Chuck, Elizabeth) = (husband, wife)
    # [Martha] and her son, [Mark], went to the park to fly a kite. They had fun doing it all day.
    relation(Martha, Mark) = (mother, son)
    # [Brandi] planned a trip to the zoo for her brother, [Michael]. They had a great time.
    relation(Brandi, Michael) = (sister, brother)
    # [Peter] asked his brother [Mark] if he would come help him fix his car next weekend.
    relation(Peter, Mark) = (brother, brother)
    # How is [Chuck] related to [Carmelita]?
    return relation(Chuck, Carmelita)
    """
    (status, result) = clutrr_satlm_exec(code.strip(), "satlm", return_code=False)

    print(status, result)
    print(code)


def test_sat():
    gts = [ "father", "grandson", "granddaughter", "mother", "mother", "granddaughter", "father-in-law", "niece"]

    with open("./temp.py", 'r') as f:
        output_code = f.read()
    examples = output_code.split('\n\n\n\n\n')

    for i, ex in enumerate(examples):
        ex = ex.split("def solution():")[1].strip()
        (status, result) = clutrr_satlm_exec(ex, "satlm",)
        print(result, gts[i])


if __name__ == "__main__":
    # ex_test()
    test_sat()

