import sys
sys.path.append('.')

from z3 import *
import re
from func_timeout import func_timeout

from prog_solver.z3_utils import timeout

def gsm_proglm_exec(completion):
    def func():
        exec(completion)
        return eval('solution()')

    return func_timeout(1, func)


def handle_variable_overwriting(lines):
    variable_regex = re.compile(r'[a-zA-Z_][a-zA-Z0-9_]*')
    new_lines = []
    num_line = len(lines)
    for i in range(num_line):
        l = lines[i]
        if '=' not in l:
            new_lines.append(l)
            continue
        eq_left, eq_right = l.split('=')
        eq_left, eq_right = eq_left.strip(), eq_right.strip()
        vars_lefts = variable_regex.findall(eq_left)
        vars_right = variable_regex.findall(eq_right)
        if not (len(vars_lefts) == 1 and vars_lefts[0] in vars_right):
            new_lines.append(l)
            continue
        var = vars_lefts[0]
        nvar = var + str(i)
        new_lines.append(f"{nvar} = {eq_right}")
        for j in range(i+1, num_line):
            lines[j] = lines[j].replace(var, nvar)
    return new_lines

def gsm_satlm_exec(code, prompting_style, return_code=False):
    lines = code.splitlines()
    lines = [line for line in lines if not line.startswith('#')]

    assert prompting_style == "satlm"

    lines = [line.strip() for line in lines]
    assert lines[-1].startswith('return ')

    lines = handle_variable_overwriting(lines)

    answer_line = lines[-1]
    answer_name = answer_line[7:]
    lines = lines[:-1]

    translated_lines = []
    declared_vars = []
    symbol_vars = []
    variable_regex = re.compile(r'[a-zA-Z_][a-zA-Z0-9_]*')

    translated_lines.append('z3._main_ctx = Context()')
    translated_lines.append('solver = Solver()')
    for line in lines:
        if "=" not in line:
            if line.strip().startswith('#') or line.strip().startswith('"""') or "def solution" in line:
                continue
            variable = line.strip()
            translated_lines.append(f'{variable} = Real(\'{variable}\')')
            if variable not in declared_vars:
                declared_vars.append(variable)
            if variable not in symbol_vars:
                symbol_vars.append(variable)
            continue
        eq_left, eq_right = line.split('=')
        eq_left, eq_right = eq_left.strip(), eq_right.strip()

        if eq_right == 'Variable()' or eq_right == 'Var()':
            if len(re.findall(variable_regex, eq_left)) > 1:
                continue
            variable = eq_left.strip()
            translated_lines.append(f'{variable} = Real(\'{variable}\')')
            if variable not in declared_vars:
                declared_vars.append(variable)
            if variable not in symbol_vars:
                symbol_vars.append(variable)
        elif not re.search(variable_regex, eq_right):
            variables = variable_regex.findall(line)
            if len(variables) == 1:
                variable = variables[0]
                if variable in symbol_vars:
                    translated_lines.append('solver.add({})'.format(line.replace('=', '==')))
                else:
                    translated_lines.append(line)
                    declared_vars.append(variables[0])
            else:
                translated_lines.append('solver.add({})'.format(line.replace('=', '==')))
        else:
            variables = variable_regex.findall(line)
            for variable in variables:
                if variable not in declared_vars:
                    translated_lines.append(f'{variable} = Real(\'{variable}\')')
                    declared_vars.append(variable)
                    symbol_vars.append(variable)
            translated_lines.append('solver.add({})'.format(line.replace('=', '==')))
    translated_lines.extend([
        'if solver.check() == unsat:',
        '    return False, "UNSAT"',
        'final_val = solver.model().eval({})'.format(answer_name),
        'solver.add({} != final_val)'.format(answer_name),
        'if solver.check() == sat:',
        '    return False, "AMBIG"',
        'return True, final_val'
    ])

    function_wrap = "def solution():\n" + "\n".join(["    " + line for line in translated_lines])
    def func():
        exec(function_wrap)
        return eval('solution()')

    status, result = func_timeout(1, func)
    if not status:
        return_val = result
    else:
        if result.is_int():
            return_val = result.as_long()
        else:
            return_val = result.as_decimal(6).rstrip('?')

    if return_code:
        return function_wrap, return_val
    else:
        return return_val


def test():

    x = """
    evan_dollars = Variable()
    markese_dollars = evan_dollars - 5
    total_dollars = evan_dollars + markese_dollars
    total_dollars = 37
    result = markese_dollars
    return result
"""
    code, res = gsm_satlm_exec(x.strip(), "satlm", return_code=True)

    print(code)
    print(res)

if __name__ == "__main__":
    test()
