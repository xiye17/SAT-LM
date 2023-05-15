import sys
sys.path.append('.')

import re
import os
import pickle

from func_timeout import func_timeout

from prog_solver.z3_utils import make_z3_enum_line, execute_z3_test

def break_down_func_var():
    pass

VAR_REGEX = r"[_a-zA-Z]+[,)]"
FUNC_REGEX = r"[_a-zA-Z]+[(]"

PREDEFIND_FUNCS = ["ForAll", "Exist", "And", "Or", "Not", "Implies"]
PREDEFIND_QUNT_VARS = ["x"]

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
    
        
def proof_satlm_exec(code, prompting_style, return_code=False):
    lines = code.splitlines()
    lines = [l.strip() for l in lines]
    lines = [l for l in lines if l and not l.startswith("#")]

    assert lines[-1].startswith("return")
    result_line = lines[-1]
    lines = lines[:-1]

    vars = set()
    functions = set()

    for line in lines:
        line_vars, line_funcs = extract_var_and_func(line)
        vars.update(line_vars)
        functions.update(line_funcs)

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

    for line in lines:
        translated_lines.append("precond.append({})".format(line))

    translated_lines.append("s = Solver()")
    translated_lines.append("s.add(precond)")

    return_clause = result_line.split("return")[1].strip()
    translated_lines.append("s.add(Not({}))".format(return_clause))
    translated_lines.extend([
        "if s.check() == unsat:",
        "    print('True')",
        "else:",
        "    print('False')",
    ])
    translated_lines = ["from z3 import *"] + translated_lines

    code = "\n".join(translated_lines)
    result = execute_z3_test(code)
    if return_code:
        return code, result
    else:
        return result


def proof_proglm_exec(code, return_code=False):
    lines = code.splitlines()
    lines = [l.strip() for l in lines]
    lines = [l for l in lines if l and not l.startswith("#")]

    function_wrap = "def solution():\n" + "\n".join(["    " + line for line in lines])
    def func():
        exec(function_wrap)
        return eval('solution()')

    result = func_timeout(1, func)
    if return_code:
        return function_wrap, result
    else:
        return result


def test_sat():
    gts = ["True", "False", "True", "False"]

    with open("temp.py") as f:
        output_code = f.read()
    examples = output_code.split('\n\n\n\n\n')

    for i, ex in enumerate(examples):
        ex = ex.split("def solution():")[1].strip()
        code, (status, result) = proof_satlm_exec(ex, "satlm", return_code=True)
        print(result, gts[i])
        # if result != gts[i]:
        #     print(code)
        #     exit()

if __name__=="__main__":
    test_sat()
