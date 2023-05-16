import sys
sys.path.append('.')

import re
import hashlib

import os
from os.path import join
import subprocess
from subprocess import check_output
from prog_solver.arlsat_parser import LSATSatProblem

PREFIX = "tmp"
def hash_of_code(code, size=16):
    val = hashlib.sha1(code.encode("utf-8")).hexdigest()
    return val[-size:]

def execution_test(code, filename=None):
    if filename is None:
        filename = hash_of_code(code)
    filename = join(PREFIX, filename + ".py")
    with open(filename, "w") as f:
        f.write(code)
    try:
        output = check_output(["python", filename], stderr=subprocess.STDOUT, timeout=1.0)
    except subprocess.CalledProcessError as e:
        output = e.output.decode("utf-8").strip().splitlines()[-1]
        result = (False, "ExecutionError " + output)
        return result
    except subprocess.TimeoutExpired:
        result = (False, "TimeoutError")
        return result
    output = output.decode("utf-8").strip()
    os.remove(filename)
    return (True,  output.splitlines())


def arlsat_satlm_exec(completion):
    try:
        code = LSATSatProblem.from_raw_statements(completion).to_standard_code()
    except:
        result = (False, "CompileError")
        return result
    return execution_test(code)


def annotation_sanity_check():
    ex_list = os.listdir("annotations/arlsat/")
    # print(ex_list)
    for ex in ex_list:
        with open(join("annotations/arlsat/", ex, "satlm.py")) as f:
            raw_statements = f.read()
        assert raw_statements.strip() == raw_statements
        problem = LSATSatProblem.from_raw_statements(raw_statements)
        std_code = problem.to_standard_code()
        print(ex)
        print(execution_test(std_code))

if __name__=="__main__":
    annotation_sanity_check()
