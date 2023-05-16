import os
import argparse
import itertools
from random import choices

from tqdm import tqdm
from math import ceil

import numpy as np

from utils import *
from api_utils import (
    run_completion_tasks_with_cache,
    config_args_and_api,
    register_base_args,
)

from task_helper import TaskHelper, load_train_test_set
from run_manual import run_evaluation, get_eval_split_abbrev
from task_evaluator import TaskEvaluator, get_task_evaluator, Prediction, print_tabular_results

def register_multistage_args(parser):
    parser.add_argument('--sig_method', type=str, default="manual", choices=["manual"])
    parser.add_argument('--sig_style_template', type=str, default="sigtpl")
    parser.add_argument('--sig_prompt_id', type=str, default="sigz3")

    parser.add_argument('--num_trans_shots', type=int, default=3)
    parser.add_argument('--trans_setting', type=str, default="setupsatlm", choices=["setupsatlm",])


SIG_STAGE = "SIG"
TRANA_STAGE = "TRANS"

TRANS_ANNOTATION_DIR = "annotations"

class MultiStageTaskHelper:
    style_to_completion_length = {}
    style_to_train_sep = {}

    def __init__(self, style):
        self.style = style

    @classmethod
    def from_taskname(cls, taskname, style):
            raise RuntimeError("Not Implemented Yet")

    def prompt_func(self, test_ex, shots):
        raise RuntimeError("Not Implemented Yet")

    def get_completion_length(self):
        return self.style_to_completion_length[self.style]

    def get_train_sep(self):
        return self.style_to_train_sep[self.style]


class SigStageHelper(MultiStageTaskHelper):
    @classmethod
    def from_taskname(cls, taskname, style):
        if taskname == "arlsat":
            return SigArLSATTaskHelper(style)
        else:
            raise RuntimeError("Not Implemented Yet")


class SigArLSATTaskHelper(SigStageHelper):
    CHOICE_IDX = ["(A)", "(B)", "(C)", "(D)", "(E)"]
    CODE_HEADER = "### write python code to answer the question"
    CODE_BLOCK_COMMENT = '"""'

    style_to_completion_length = {
        "sigtpl": 256,
    }

    style_to_train_sep = {
        "sigtpl": "\n\n\n\n",
    }

    def prompt_func(self, test_ex, shots):
        if self.style == "sigtpl":
            return self.sigtpl_prompt(test_ex, shots)
        else:
            raise RuntimeError("Not Implemented Yet")

    def _single_ex_func(self, ex, is_train):
        assert not is_train
        choice_str = "\n".join([self.CHOICE_IDX[i] + " " + x for (i, x) in enumerate(ex["choices"])])
        p_ex = "{}\n{}\n{}\nQuestion: {}\nChoices:\n{}\n{}\n".format(
            self.CODE_HEADER,
            self.CODE_BLOCK_COMMENT,
            ex["context"], ex["question"], choice_str,
            self.CODE_BLOCK_COMMENT)
        return p_ex

    def sigtpl_prompt(self, test_ex, shots):
        showcase_examples = [
            self._single_ex_func(s, True) for s in shots
        ]
        test_example = [self._single_ex_func(test_ex, False)]
        return  self.get_train_sep().join(showcase_examples + test_example)


class TransStageHelper(MultiStageTaskHelper):
    @classmethod
    def from_taskname(cls, taskname, style):
        if taskname == "arlsat":
            return TransArLSATTaskHelper(style)
        else:
            raise RuntimeError("Not Implemented Yet")


class TransArLSATTaskHelper(SigStageHelper):
    CHOICE_IDX = ["(A)", "(B)", "(C)", "(D)", "(E)"]
    CODE_HEADER = "### write python code to answer the question"
    CODE_BLOCK_COMMENT = '"""'

    style_to_completion_length = {
        "transtpl": 256,
    }

    style_to_train_sep = {
        "transtpl": "\n\n\n\n",
    }

    def prompt_func(self, test_ex, shots):
        if self.style == "transtpl":
            return self.transtpl_prompt(test_ex, shots)
        else:
            raise RuntimeError("Not Implemented Yet")

    def _single_ex_func(self, ex, is_train):
        assert not is_train
        choice_str = "\n".join([self.CHOICE_IDX[i] + " " + x for (i, x) in enumerate(ex["choices"])])
        p_ex = "{}\n{}\n{}\nQuestion: {}\nChoices:\n{}\n{}\n".format(
            self.CODE_HEADER,
            self.CODE_BLOCK_COMMENT,
            ex["context"], ex["question"], choice_str,
            self.CODE_BLOCK_COMMENT)
        return p_ex

    def transtpl_prompt(self, test_ex, shots):
        showcase_examples = [
            self._single_ex_func(s, True) for s in shots
        ]
        test_example = [self._single_ex_func(test_ex, False)]
        return  self.get_train_sep().join(showcase_examples + test_example)


class SignatureInfo:
    def __init__(self, completion, style_template):
        self.completion = completion
        self.style_template = style_template
        self.keywords = self.extract_keywords(completion)


    def extract_keywords(self, completion):
        lines = [x.strip() for x in completion.split("\n")]
        decl_lines = [x for x in lines if "EnumSort" in x or "Function" in x]
        print_lines = [x for x in lines if "print" in x]
        question_line = next((x for x in lines if "# Question" in x), "")

        keywords = set()

        enum_types = {}
        for line in decl_lines:
            if "EnumSort" in line:
                sort_name = line.split("=")[0].strip()
                sort_member_str = line.split("=")[1].strip()[len("EnumSort("):-1]
                sort_members = [x.strip() for x in sort_member_str[1:-1].split(",")]
                if all([x.isdigit() for x in sort_members]):
                    enum_types[sort_name] = "EnumInt"
                    keywords.add(enum_types[sort_name])
                else:
                    enum_types[sort_name] = "EnumVal"

            elif "Function" in line:
                function_args_str = line.split("=")[1].strip()[len("Function("):-1]
                function_args = [x.strip() for x in function_args_str.split(",")]
                function_sig = [enum_types[x] if x in enum_types else x for x in function_args]
                function_sig = "(" + ",".join(function_sig) + ")"
                function_sig = function_sig.replace("EnumInt", "int")
                if "int" in function_sig:
                    keywords.add("int")
                if "bool" in function_sig:
                    keywords.add("bool")
                keywords.add(function_sig)
            else:
                raise RuntimeError("Unknown declaration statement: {}".format(line))

        if " if " in question_line.lower():
            keywords.add("if_question")

        for line in print_lines:
            line = line[len("print("):-1]
            if "exception" in line:
                keywords.add("exception")
                line = line[len("exception("):-1]
            keywords.add(line.strip()[:-2])
        return keywords


class TransSetting:
    SETTING_TO_MATHOD = {
        "setupsatlm": {
            "question_style": "satlm",
            "selection": "signature",
            "prompt": "satlm",
            "train_sep": "\n\n\n\n",
            "completion_length": 1024,
        },
    }

    def __init__(self, args):
        self.args = args
        setting_version = args.trans_setting
        self.setting_version = setting_version
        self.setting = self.SETTING_TO_MATHOD[setting_version]

    def get_style_template(self):
        return self.setting["question_style"]

    def get_train_sep(self):
        return self.setting["train_sep"]

    def get_completion_length(self):
        return self.setting["completion_length"]

    def shot_selection(self, test_signature, train_signatures, num_shots):
        if self.setting["selection"] == "signature":
            return self.signature_base_shots_selection(test_signature, train_signatures, num_shots)
        else:
            raise RuntimeError("Not Implemented Yet")

    def construct_prompt(self, test_ex, train_annotations):
        if self.setting["prompt"] == "satlm":
            return self.predefined_prompt(self.setting["prompt"], test_ex, train_annotations)
        else:
            raise RuntimeError("Not Implemented Yet")

    def encode_question(self, test_ex):
        if self.setting["question_style"] == "satlm":
            return self.satlm_encode_question(test_ex)
        else:
            raise RuntimeError("Not Implemented Yet")

    def satlm_encode_question(self, ex):
        CHOICE_IDX = ["(A)", "(B)", "(C)", "(D)", "(E)"]
        CODE_HEADER = "### write python code to answer the question"
        CODE_BLOCK_COMMENT = '"""'
        choice_str = "\n".join([CHOICE_IDX[i] + " " + x for (i, x) in enumerate(ex["choices"])])
        p_ex = "{}\n{}\n{}\nQuestion: {}\nChoices:\n{}\n{}\n".format(
            CODE_HEADER,
            CODE_BLOCK_COMMENT,
            ex["context"], ex["question"], choice_str,
            CODE_BLOCK_COMMENT
        )
        return p_ex

    def predefined_prompt(self, predev_version, test_ex, train_annotations):
        showcase_examples = [x[predev_version] for x in train_annotations]
        test_example = [self.encode_question(test_ex)]
        return  self.get_train_sep().join(showcase_examples + test_example)

    # return indexes of the shot
    def signature_base_shots_selection(self, test_signature, train_signatures, num_shots):
        # try to cover as many keywords as possible
        full_keywords = set(test_signature.keywords)
        remaining_keywords = set(test_signature.keywords)

        selected_indexes = []
        for _ in range(num_shots):
            # max_full_gain = (-1, -1)
            # max_rem_gain = (-1, -1)
            max_gain = ((-1, -1, -1, -1), -1)
            for i, train_signature in enumerate(train_signatures):
                if i in selected_indexes:
                    continue
                rem_gain = len(remaining_keywords.intersection(train_signature.keywords))
                rem_gain_ratio = rem_gain / len(train_signature.keywords)
                full_gain = len(full_keywords.intersection(train_signature.keywords))
                full_gain_ratio = full_gain / len(train_signature.keywords)
                comp_key = (rem_gain, rem_gain_ratio, full_gain, full_gain_ratio)
                if comp_key >= max_gain[0]:
                    max_gain = (comp_key, i)

            selected_indexes.append(max_gain[1])
            remaining_keywords = remaining_keywords.difference(train_signatures[max_gain[1]].keywords)

        return selected_indexes

def read_manual_prompt(task, stage, prompt_id, style_template):    
    prompt_lines = read_jsonline(f'manual_prompts/multistage_{task}.jsonline')
    d = dict([(x["id"], x) for x in prompt_lines])
    selected = d[prompt_id]
    assert selected["style_template"] == style_template
    return selected["prompt"]


def sig_stage_result_filename_func(args):
    if args.sig_method == "manual":
        prompt_id = "manual" + args.sig_prompt_id
    else:
        raise RuntimeError("Not Implemented Yet")

    return "misc/multisgate-sig-{}--eng{}--{}{}-{}--{}--numsamp{}--temp{}--sty{}--predictions.json".format(
        args.task,
        args.engine,
        get_eval_split_abbrev(args),
        args.slice_dev, args.slice_dev + args.num_dev,
        prompt_id,
        args.num_samples,
        args.temperature,
        args.sig_style_template
    )

def trans_stage_result_filename_func(args):
    if args.sig_method == "manual":
        sig_p_id = "manual" + args.sig_prompt_id
    else:
        raise RuntimeError("Not Implemented Yet")
    return "misc/multisgate-trans-{}--eng{}--{}{}-{}--sig{}--st{}--{}--numsamp{}--temp{}--sty{}--predictions.json".format(
        args.task,
        args.engine,
        get_eval_split_abbrev(args),
        args.slice_dev, args.slice_dev + args.num_dev,
        sig_p_id,
        args.trans_setting,
        args.num_trans_shots,
        args.num_samples,
        args.temperature,
        args.sig_style_template
    )


def parse_problem_signatures(args, responses, task_helper):
    signatures = []
    for reps in responses:
        sigs = []
        for r in reps:
            completion = r["text"].strip()
            sig = SignatureInfo(completion, args.sig_style_template)
            sigs.append(sig)

        signatures.append(sigs)

    return signatures

def run_signature_stage(args, test_data):
    task_helper = SigStageHelper.from_taskname(args.task, args.sig_style_template)

    # construct signature prompt
    if args.sig_method == "manual":
        base_manual_prompt = read_manual_prompt(args.task, SIG_STAGE, args.sig_prompt_id, args.sig_style_template)
    else:
        raise RuntimeError("Not Implemented Yet")

    prompts_to_complete = []    
    for test_ex in test_data:
        test_part = task_helper.prompt_func(test_ex, [])
        
        prompts_to_complete.append(
            [base_manual_prompt + task_helper.get_train_sep() + test_part]
        )

    _batch_size, _temperature, _num_samples = args.batch_size, args.temperature, args.num_samples
    args.batch_size, args.temperature, args.num_samples = 5, 0.0, 1
    task_max_tokens = task_helper.get_completion_length()
    task_stop_token = task_helper.get_train_sep()
    cache_filename = sig_stage_result_filename_func(args)
    responses = run_completion_tasks_with_cache(args, cache_filename, prompts_to_complete, task_max_tokens, task_stop_token)
    responses = [flatten_nested_list(resps_by_example) for resps_by_example in responses]
    args.batch_size, args.temperature, args.num_samples = _batch_size, _temperature, _num_samples


    # signature stage evaluation
    problem_signatures = parse_problem_signatures(args, responses, task_helper)
    return problem_signatures


TASK_ANNOTATION_DICT = {
    "arlsat": ["signature", "satlm",],
}

def read_trans_annotations(args):
    prefix = join(TRANS_ANNOTATION_DIR, args.task)

    annotation_list = TASK_ANNOTATION_DICT[args.task]

    annotations = []
    ex_names = [x for x in os.listdir(prefix) if not x.startswith(".")]
    ex_names = sorted(ex_names, key=lambda x: int(re.findall(r"\d+", x)[-1]))

    for ex_name in ex_names:
        if ex_name.startswith("."):
            continue
        anno = {}
        anno["name"] = ex_name
        for fname in annotation_list:
            if os.path.exists(join(prefix, ex_name, fname + ".py")):
                with open(join(prefix, ex_name, fname + ".py")) as f:
                    anno[fname] = f.read()
            else:
                anno[fname] = None
        annotations.append(anno)

    return annotations


def strip_question_head(x):
    return x.split('"""')[-1].strip()


def run_translation_stage(args, test_data, problem_signatures):
    sig_helper = SigStageHelper.from_taskname(args.task, args.sig_style_template)
    trans_setting = TransSetting(args)

    print("RUN TRANSLATION STAGE")
    train_example_annotations = read_trans_annotations(args)
    for ex_ann in train_example_annotations:
        ex_ann["sig_info"] = SignatureInfo(strip_question_head(ex_ann["signature"]), args.sig_style_template)

    prompts_to_complete = []


    for test_ex, test_sigs in zip(test_data, problem_signatures):
        prompts_for_ex = []
        for test_sig in test_sigs:
            selected_indexes = trans_setting.shot_selection(test_sig, [x["sig_info"] for x in train_example_annotations], args.num_trans_shots)
            selected_annotations = [train_example_annotations[i] for i in selected_indexes]

            prompt = trans_setting.construct_prompt(test_ex, selected_annotations)
            prompts_for_ex.append(prompt)

        prompts_to_complete.append(prompts_for_ex)
    # exit()

    task_max_tokens = trans_setting.get_completion_length()
    task_stop_token = trans_setting.get_train_sep()
    cache_filename = trans_stage_result_filename_func(args)
    responses = run_completion_tasks_with_cache(args, cache_filename, prompts_to_complete, task_max_tokens, task_stop_token)
    responses = [flatten_nested_list(resps_by_example) for resps_by_example in responses]

    args.style_template = trans_setting.get_style_template()
    eval_results = run_evaluation(args, test_data, responses)
    print_tabular_results("VOTE"+str(args.num_eval_samples), eval_results)


def multistage_prompting(args):
    _, test_data = load_train_test_set(args)

    problem_signatures = run_signature_stage(args, test_data)
    run_translation_stage(args, test_data, problem_signatures)

def main():
    parser = argparse.ArgumentParser()
    register_base_args(parser)
    register_multistage_args(parser)

    args = parser.parse_args()
    assert args.task is not None

    config_args_and_api(args)
    multistage_prompting(args)

if __name__=="__main__":
    main()
