import os
import argparse
import sys
import json
import re
import random
import func_timeout
from func_timeout import FunctionTimedOut
import numpy as np
from tqdm import tqdm

from abc import ABC
from collections import namedtuple, Counter, OrderedDict
from os.path import join

from prog_solver.gsm_solver import  gsm_proglm_exec, gsm_satlm_exec
from prog_solver.clutrr_solver import clutrr_proglm_exec, clutrr_satlm_exec
from prog_solver.proof_solver import proof_proglm_exec, proof_satlm_exec
from prog_solver.arlsat_solver import arlsat_satlm_exec


EVALUATOR_REGISTRY = {}

Prediction = namedtuple('Prediction', ['completion', 'prompt', 'logprob', 'norm_logprob'])

def print_tabular_results(row_id, eval_result):
    num_contents = [ "%.2f" % (eval_result["accuracy"] * 100), "%.2f" % (eval_result["consistency"] * 100),
        str(eval_result["avg_logprob"]), str(eval_result["avg_normlogprob"])]
    print("\t".join(["TABINFO", str(row_id)] + num_contents))

class TaskEvaluator(ABC):
    do_printing = False
    do_impose_prediction = False
    do_voting = False
    NULL_ANSWER = "NULL"
    EXCEPTION = "EXCEPTION"
    TIMEOUT = "TIMEOUT"
    AMBIG = "AMBIG"
    UNSAT = "UNSAT"

    @classmethod
    def get_task_name(cls):
        [task_name] = re.match("(.+)Evaluator", cls.__name__).groups()
        return task_name.lower()

    def __init_subclass__(cls, **kwargs):
        """Register all children in registry"""
        super().__init_subclass__(**kwargs)
        if cls == TaskEvaluator:
            # print(f"{cls} is abstract!")
            return
        task_name = cls.get_task_name().lower()
        EVALUATOR_REGISTRY[task_name] = cls

    @classmethod
    def process_instance(cls, pred, ref, prompting_style=None):
        choices = []
        gt = cls.postprocess_ground_truth(ref["label"])
        null_ans = cls.NULL_ANSWER
        prompt = pred[0].prompt
        for p in pred:
            single_comp, single_exp, single_ans = cls.parse_explanation_answer_from_completion(p.completion, prompting_style)
            choices.append({
                "completion": single_comp,
                "answer": single_ans,
                "explanation": single_exp,
                "norm_logprob": p.norm_logprob,
                "sum_logprob": p.logprob,
                "acc": str(gt == single_ans),
            })

        return {
            "prompt": prompt,
            "ground_truth": gt,
            "null_answer": null_ans,
            "completions": choices
        }

    @classmethod
    def enter_evaluation(cls):
        pass

    @classmethod
    def exit_evaluation(cls):
        pass

    @classmethod
    def generate_random_answer(cls):
        raise NotImplementedError

    @classmethod
    def evaluate(cls, predictions, examples, prompting_style=None, train_sep="\n\n", return_verbose=False):
        if isinstance(predictions[0], list) and len(predictions[0]) > 1:
            cls.do_voting = True

        cls.enter_evaluation()

        acc_records = []
        cov_records = []
        cons_records = []

        all_proced_answers = []
        all_proced_gts = []
        all_voted_answers = []

        for idx, (pred, ref) in tqdm(enumerate(zip(predictions, examples)), total=max(len(predictions), len(examples)), desc="Evaluating"):
            if isinstance(pred, list):
                all_answers = []
                comp = []
                prompt = cls.postprocess_prompt(pred[0].prompt, train_sep)
                answer_counter = {}
                for p in pred:
                    single_comp, single_ans = cls.postprocess_completion(p.completion, prompting_style, train_sep, example=ref)
                    all_answers.append(single_ans)
                    comp.append(single_comp)
                    if single_ans not in answer_counter:
                        answer_counter[single_ans] = {
                            "count": 0,
                            "max_logprob": -1e6,
                            "max_norm_logprob": -1e6,
                        }
                    stat = answer_counter[single_ans]
                    stat["count"] = stat["count"] + 1
                    stat["max_logprob"] = max(stat["max_logprob"], p.logprob)
                    stat["max_norm_logprob"] = max(stat["max_norm_logprob"], p.norm_logprob)

                sorted_answers = sorted(answer_counter.keys(), key=lambda x: (answer_counter[x]["count"], answer_counter[x]["max_norm_logprob"]), reverse=True)
                # sorted_answers = sorted(answer_counter.keys(), key=lambda x: ( answer_counter[x]["max_norm_logprob"],answer_counter[x]["count"] ), reverse=True)
                answer = sorted_answers[0]
                if answer == cls.NULL_ANSWER and len(sorted_answers) > 1:
                    answer = sorted_answers[1]
                if cls.NULL_ANSWER in sorted_answers:
                    sorted_answers.remove(cls.NULL_ANSWER)
                cons = answer_counter[answer]['count'] / len(pred)
                answer_counter = OrderedDict([(k, answer_counter[k]) for k in sorted_answers])
            else:
                prompt = cls.postprocess_prompt(pred.prompt)
                comp, answer = cls.postprocess_completion(pred.completion, prompting_style, train_sep, example=ref)
                cons = 1.0
                answer_counter = None
                all_answers = [answer]
            if answer == cls.NULL_ANSWER and cls.do_impose_prediction:
                answer = cls.generate_random_answer()

            gt = cls.postprocess_ground_truth(ref["label"])
            acc_records.append(cls.answer_equal(answer, gt, example=ref))
            cons_records.append(cons)
            if answer_counter is not None:
                cov_records.append(gt in answer_counter)
            all_proced_answers.append(all_answers)
            all_voted_answers.append(answer)
            all_proced_gts.append(gt)
            cls.print_instance_outputs(idx, prompt, comp, answer, gt, ref, answer_counter)

        eval_results = {}
        acc_records = np.array(acc_records)
        print("ACC: {:.2f}".format(np.mean(acc_records) * 100))
        print("CONS: {:.2f}".format(np.mean(cons_records) * 100))
        eval_results["accuracy"] = np.mean(acc_records)
        eval_results["consistency"] = np.mean(cons_records)
        if cov_records:
            cov_records = np.array(cov_records)
            print("COV: {:.2f}".format(np.mean(cov_records) * 100))
            eval_results["converage"] = np.mean(cov_records)
        if return_verbose:
            eval_results["all_raw_predictions"] = all_proced_answers
            eval_results["all_gts"] = all_proced_gts
            eval_results["all_voted_predictions"] = all_voted_answers
        eval_results["num"] = len(acc_records)

        cls.exit_evaluation()
        return eval_results

    @staticmethod
    def answer_equal(pred, gt, example=None):
        return pred == gt

    @classmethod
    def print_instance_outputs(cls, idx, prompt, comp, answer, gt, ref, answer_counter=None):
        if cls.do_printing:
            print("\n---------------------------------------------")
            print("Prompt:", prompt)
            if isinstance(comp, list):
                print("Completion:")
                for c in comp[:1]:
                    print("\t" + c.strip())
                if answer_counter:
                    print("\tCounter:", answer_counter)
            else:
                print("Completion:", comp.strip())
            print("Answer:", answer, " | GT:", gt)
            if not cls.do_voting:
                print("IDX:", idx, "ACC:", cls.answer_equal(answer, gt, example=ref), "ANS:", answer)
            elif answer_counter:
                # get the frequency of most frequent answer
                if answer == cls.NULL_ANSWER:
                    max_freq = 0
                    max_cons = 0
                else:
                    assert cls.NULL_ANSWER not in answer_counter
                    values = [x["count"] for x in answer_counter.values()]
                    max_freq = max(values)
                    max_cons = max_freq / sum(values)
                print("IDX:", idx, "ACC:", cls.answer_equal(answer, gt, example=ref), "ANS:", answer, "FREQ:", max_freq, "CONS:", max_cons)
       
    @classmethod
    def core_evaluation(cls, predictions, examples, prompting_style=None):
        raise NotImplementedError()

    # process completion, return processed completion and answer
    @staticmethod
    def postprocess_completion(completion, prompting_style, train_sep, example=None):
        raise NotImplementedError()

    @staticmethod
    def postprocess_ground_truth(gt):
        raise NotImplementedError()

    @classmethod
    def parse_explanation_answer_from_completion(cls, completion, prompting_style):
        raise NotImplementedError()

    @staticmethod
    def postprocess_prompt(prompt, train_sep):
        return prompt.split(train_sep)[-1].strip()



class GSMEvaluator(TaskEvaluator):
    ANSWER_RE = re.compile(r"(\-?[0-9\.\,]+)")
    ANSWER_HINT = "the answer is"

    GSM_ERROR_ANSWER = [
        TaskEvaluator.UNSAT, TaskEvaluator.EXCEPTION, TaskEvaluator.TIMEOUT, TaskEvaluator.AMBIG
    ]

    @staticmethod
    def postprocess_ground_truth(gt):
        try:
            return float(GSMEvaluator.extract_answer(gt).strip())
        except:
            return GSMEvaluator.NULL_ANSWER

    @staticmethod
    def answer_equal(pred, gt, example=None):
        if pred == GSMEvaluator.NULL_ANSWER or gt == GSMEvaluator.NULL_ANSWER:
            return False
        if isinstance(pred, str):
            return False
        return  abs(pred - gt) < 1e-3

    @staticmethod
    def postprocess_completion(completion, prompting_style, train_sep, example=None):
        completion = completion.rstrip().split(train_sep)[0]
        if prompting_style == "cot" or prompting_style == "std" or prompting_style == "satcotsolver":
            return GSMEvaluator.postprocess_qa_style_completion(completion)
        elif prompting_style == "proglm":
            return GSMEvaluator.postprocess_prog_style_completion(completion)
        elif prompting_style == "satlm":
            return GSMEvaluator.postprocess_sat_style_completion(completion, prompting_style)
        else:
            raise RuntimeError("Not implemented")

    @staticmethod
    def postprocess_qa_style_completion(completion):
        hint_sent = "the answer is"
        completion_lower = completion.lower()
        if hint_sent in completion_lower:
            answer = completion_lower.split(hint_sent)[1].rstrip(".").strip()
        else:
            answer = completion_lower
        numeric_answer = GSMEvaluator.extract_answer(answer).strip()
        try:
            numeric_answer = float(numeric_answer)
        except ValueError:
            numeric_answer = GSMEvaluator.NULL_ANSWER
        return completion, numeric_answer

    @staticmethod
    def postprocess_prog_style_completion(completion):

        try:
            answer = gsm_proglm_exec(completion)
            answer = float(answer)
        except FunctionTimedOut:
            answer = TaskEvaluator.TIMEOUT
        except Exception as e:
            answer = TaskEvaluator.EXCEPTION

        if GSMEvaluator.do_voting:
            if answer in GSMEvaluator.GSM_ERROR_ANSWER:
                answer = GSMEvaluator.NULL_ANSWER
        return completion, answer

    @staticmethod
    def postprocess_sat_style_completion(completion, prompting_style):
        try:
            answer = gsm_satlm_exec(completion, prompting_style)
            if answer == TaskEvaluator.UNSAT or answer == TaskEvaluator.AMBIG:
                pass
            else:
                answer = float(answer)
        except FunctionTimedOut as e:
            answer = TaskEvaluator.TIMEOUT
        except Exception as e:
            answer = TaskEvaluator.EXCEPTION

        if GSMEvaluator.do_voting:
            if answer in GSMEvaluator.GSM_ERROR_ANSWER:
                answer = GSMEvaluator.NULL_ANSWER
        return completion, answer

    @staticmethod
    def extract_answer(completion):
        match = GSMEvaluator.ANSWER_RE.search(completion)
        if match:
            match_str = match.group(0).strip()
            match_str = match_str.replace(",", "")
            return match_str
        else:
            return GSMEvaluator.NULL_ANSWER

class CLUTRREvaluator(TaskEvaluator):
    @staticmethod
    def postprocess_ground_truth(gt):
        return gt.strip()

    @staticmethod
    def postprocess_completion(completion, prompting_style, train_sep, example=None):
        completion = completion.rstrip().split(train_sep)[0]
        if prompting_style == "proglm":
            return CLUTRREvaluator.postprocess_prog_style_completion(completion)
        elif prompting_style == "satlm":
            return CLUTRREvaluator.postprocess_sat_style_completion(completion, prompting_style)
        elif prompting_style == "satcotsolver":
            return CLUTRREvaluator.postprocess_qa_style_completion(completion, prompting_style)
        else:
            raise RuntimeError("Not implemented")

    @staticmethod
    def postprocess_qa_style_completion(completion, prompting_style):
        hint_sent = "the answer is"
        completion_lower = completion.lower()
        if hint_sent in completion_lower:
            answer = completion_lower.split(hint_sent)[1].rstrip(".").strip()
        else:
            answer = CLUTRREvaluator.NULL_ANSWER
        return completion, answer

    @staticmethod
    def postprocess_prog_style_completion(completion):
        try:
            result = clutrr_proglm_exec(completion)
        except:
            result = CLUTRREvaluator.EXCEPTION

        if CLUTRREvaluator.do_voting:
            if result == CLUTRREvaluator.EXCEPTION:
                result = CLUTRREvaluator.NULL_ANSWER

        return completion, result


    @staticmethod
    def postprocess_sat_style_completion(completion, prompting_style):
        try:
            completion = completion.split("def solution():")[1].strip()
            (status, result) = clutrr_satlm_exec(completion, prompting_style)
            if not status:
                result = CLUTRREvaluator.EXCEPTION
        except:
            result = CLUTRREvaluator.EXCEPTION

        if CLUTRREvaluator.do_voting:
            if result in [CLUTRREvaluator.AMBIG, CLUTRREvaluator.EXCEPTION,
                            CLUTRREvaluator.UNSAT, CLUTRREvaluator.TIMEOUT]:
                result = CLUTRREvaluator.NULL_ANSWER
        return completion, result

class ProofD5Evaluator(TaskEvaluator):
    @staticmethod
    def postprocess_ground_truth(gt):
        return str(gt)

    @classmethod
    def enter_evaluation(cls):
        random.seed(42)

    @staticmethod
    def postprocess_completion(completion, prompting_style, train_sep, example=None):
        completion = completion.rstrip().split(train_sep)[0]
        if prompting_style == "cot" or prompting_style == "std":
            return ProofD5Evaluator.postprocess_cot_style_completion(completion)
        elif prompting_style == "satlm":
            return ProofD5Evaluator.postprocess_sat_style_completion(completion, prompting_style)
        elif prompting_style == "proglm":
            return ProofD5Evaluator.postprocess_prog_style_completion(completion)
        else:
            raise RuntimeError("Not implemented")

    @staticmethod
    def postprocess_cot_style_completion(completion):
        if "the statement is" in completion:
            result = completion.strip().split("the statement is ")[1].strip().rstrip(".")
        else:
            result = ProofD5Evaluator.NULL_ANSWER

        return completion, result

    @staticmethod
    def postprocess_sat_style_completion(completion, prompting_style):
        completion = completion.strip()
        try:
            _, result = proof_satlm_exec(completion, prompting_style)
            result = result.strip()
        except:
            result = ProofD5Evaluator.NULL_ANSWER
        return completion, result

    @staticmethod
    def postprocess_prog_style_completion(completion):
        completion = completion.strip()
        try:
            result = proof_proglm_exec(completion)
            result = str(result)
        except:
            result = ProofD5Evaluator.NULL_ANSWER

        return completion, result

    @classmethod
    def generate_random_answer(cls):
        return random.choice(["True", "False"])


class LongContextMCEvaluator(TaskEvaluator):
    ANSWER_HINT = "the answer is"
    CHOICES = ['a', 'b', 'c', 'd', 'e']

    @staticmethod
    def postprocess_ground_truth(gt):
        return gt

    @classmethod
    def enter_evaluation(cls):
        random.seed(42)

    @classmethod
    def generate_random_answer(cls):
        return random.choice([0, 1, 2, 3])

    @staticmethod
    def postprocess_completion(completion, prompting_style, train_sep, example=None):
        completion = completion.rstrip().split(train_sep)[0]
        if prompting_style == "std" or prompting_style == "cot":
            return LongContextMCEvaluator.postprocess_qa_style_completion(completion)
        elif prompting_style == "satlm":
            return LongContextMCEvaluator.postprocess_sat_style_completion(completion)
        else:
            raise RuntimeError("Not implemented")

    @staticmethod
    def postprocess_sat_style_completion(completion):
        status, result = arlsat_satlm_exec(completion)
        if not status:
            result = LongContextMCEvaluator.NULL_ANSWER
        else:
            if len(result) == 0:
                result = LongContextMCEvaluator.NULL_ANSWER
            else:
                answer = result[-1].lower().rstrip(".").strip()
                answer = answer.lstrip('(').rstrip(')')
                if answer in LongContextMCEvaluator.CHOICES:
                    answer = LongContextMCEvaluator.CHOICES.index(answer)
                else:
                    answer = LongContextMCEvaluator.NULL_ANSWER
                result = answer
        return completion, result

    @staticmethod
    def postprocess_qa_style_completion(completion):
        hint_sent = "the answer is"
        completion_lower = completion.lower()
        if hint_sent in completion_lower:
            answer = completion_lower.split(hint_sent)[1].rstrip(".").strip()
            answer = answer.lstrip('(').rstrip(')')
            if answer in LongContextMCEvaluator.CHOICES:
                answer = LongContextMCEvaluator.CHOICES.index(answer)
            else:
                answer = LongContextMCEvaluator.NULL_ANSWER
        else:
            answer = LongContextMCEvaluator.NULL_ANSWER
        return completion, answer


class ArLSATEvaluator(LongContextMCEvaluator):
    pass


def get_task_evaluator(taskname):
    return EVALUATOR_REGISTRY[taskname.lower()]

