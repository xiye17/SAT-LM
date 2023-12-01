import sys
sys.path.append('.')

import re
import os
import pickle

from func_timeout import func_timeout
from prog_solver.z3_utils import make_z3_enum_line, execute_z3_test

from utils import read_jsonline

def break_down_func_var():
    pass

VAR_REGEX = r"[_a-zA-Z]+[,)]"
FUNC_REGEX = r"[_a-zA-Z][_a-zA-Z0-9]+[(]"

RULE_REGEX = r"Rule[0-9]+"

PREDEFIND_FUNCS = ["ForAll", "Exists", "And", "Or", "Not", "Implies"]
PREDEFIND_QUNT_VARS = ["x"]

ENUM_MAX_SAT_FUNC = """
def enumnerate_max_sat(precond, soft_rules, question):
    n_soft = len(soft_rules)
    for i in range(n_soft + 1):
        for c in combinations(soft_rules, n_soft - i):
            s = Solver()
            s.add(precond)
            s.add(c)
            pos_side = s.check(question)
            neg_side = s.check(Not(question))
            if pos_side == sat and neg_side == unsat:
                return "yes"
            elif pos_side == unsat and neg_side == sat:
                return "no"
            elif pos_side == sat and neg_side == sat:
                return "unknown"
    return "unsat"
""".strip()

def extract_var_and_func(line):
    all_vars = re.findall(VAR_REGEX, line)
    all_funcs = re.findall(FUNC_REGEX, line)
    all_vars = [all_vars.rstrip(",)") for all_vars in all_vars]
    all_funcs = [all_funcs.rstrip("(") for all_funcs in all_funcs]
    return all_vars, all_funcs

def determine_func_n_args(code, func):
    start_pos = code.find(func + "(")
    end_pos = code.find(")", start_pos)
    num_args = code[start_pos+len(func)+1:end_pos].count(",") + 1
    return num_args

def board_satlm_exec(code, return_code=False):
    lines = code.splitlines()
    lines = [l.strip() for l in lines]
    lines = [l for l in lines if l and not l.startswith("#")]

    assert lines[-1].startswith("return")
    result_line = lines[-1]
    lines = lines[:-1]

    vars = set()
    functions = set()
    rules = set()
    for line in lines:
        line_vars, line_funcs = extract_var_and_func(line)
        vars.update(line_vars)
        functions.update(line_funcs)
        rules.update(re.findall(RULE_REGEX, line))

    vars = [v for v in vars if v not in PREDEFIND_QUNT_VARS]
    functions = [f for f in functions if f not in PREDEFIND_FUNCS]


    func_n_args = {}
    for func in functions:
        func_n_args[func] = determine_func_n_args(code, func)
    functions = sorted(functions, key=lambda x: func_n_args[x])

    translated_lines = []
    translated_lines.append(make_z3_enum_line("ThingsSort", vars))

    for func in functions:
        num_args = func_n_args[func]
        translated_lines.append("{} = Function('{}', {}, BoolSort())".format(func, func, ", ".join(["ThingsSort"]*num_args)))
    translated_lines.append("x = Const('x', ThingsSort)")
    translated_lines.append("precond = []")

    rule_def_lines = []
    rule_rev_lines = []
    fact_def_lines = []
    soft_rule_line = None
    for line in lines:
        if line.startswith("soft_rules"):
            soft_rule_line = line
            continue
        if re.match(RULE_REGEX, line):
            if soft_rule_line:
                rule_rev_lines.append(line)
            else:
                rule_def_lines.append(line)
        else:
            fact_def_lines.append(line)
    for line in rule_def_lines:
        translated_lines.append(line)
    translated_lines.append(soft_rule_line)
    for line in rule_rev_lines:
        translated_lines.append(line)
    for line in fact_def_lines:
        translated_lines.append("precond.append({})".format(line))

    for rule in rules:
        translated_lines.append("precond.append({})".format(rule))
    return_clause = result_line.split("return")[1].strip()
    translated_lines.append("question = {}".format(return_clause))

    translated_lines.extend(ENUM_MAX_SAT_FUNC.splitlines())
    translated_lines = ["from z3 import *", "from itertools import combinations"] + translated_lines
    translated_lines.append("print(enumnerate_max_sat(precond, soft_rules, question))")
    code = "\n".join(translated_lines)
    result = execute_z3_test(code)
    if return_code:
        return code, result
    else:
        return result

def read_manual_prompt(task, prompt_id, style_template):    
    prompt_lines = read_jsonline(f'manual_prompts/{task}.jsonline')
    d = dict([(x["id"], x) for x in prompt_lines])
    selected = d[prompt_id]
    assert selected["style_template"] == style_template
    return selected["prompt"]

def test_main_sat():
    gts = ["unknown", "yes", "no", "yes", "no", "unknown", "unknown"]
    output_code = read_manual_prompt("boardmaindp1", "satlm", "satlm")
    examples = output_code.split('\n\n\n\n\n')
    for i, ex in enumerate(examples):
        ex = ex.split("def solution():")[1].strip()
        code, (status, result) = board_satlm_exec(ex, return_code=True)
        print(result, gts[i])

if __name__=="__main__":
    test_main_sat()
