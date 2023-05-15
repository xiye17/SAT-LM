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
    score_of_completion,
    confidence_of_completion
)

from task_helper import TaskHelper, load_train_test_set
from task_evaluator import TaskEvaluator, get_task_evaluator, Prediction, print_tabular_results

def get_eval_split_abbrev(args):
    return args.eval_split

def run_evaluation(args, test_data, responses, print_perplexity=True, return_verbose=False):
    evaluator = get_task_evaluator(args.task)

    prompting_style = args.style_template
    task_helper = TaskHelper.from_taskname(args.task, args.style_template)

    max_sample_num = max([len(x) for x in responses]) if responses else 0
    num_eval_samples = args.num_eval_samples if args.num_eval_samples > 0 else max_sample_num
    if args.first_k > 0:
        test_data = test_data[:args.first_k]
        responses = responses[:args.first_k]

    predictions = [
        [Prediction(x["text"], x["prompt"], *score_of_completion(x)) for x in completions[:num_eval_samples]] for completions in responses
    ]

    if args.do_print:
        TaskEvaluator.do_printing = True
    if args.do_impose_prediction:
        TaskEvaluator.do_impose_prediction = True

    sums = np.array([[x.logprob for x in preds] for preds in predictions])
    norms = np.array([[x.norm_logprob for x in preds] for preds in predictions])
    avg_sum = sums.mean(axis=1).mean(axis=0)
    avg_norm = norms.mean(axis=1).mean(axis=0)

    if print_perplexity:
        print("AVG Logprob: {:.4f}".format(avg_sum))
        print("AVG Norm Logprob: {:.4f}".format(avg_norm))

    eval_results = evaluator.evaluate(predictions, test_data, prompting_style, train_sep=task_helper.get_train_sep(), return_verbose=return_verbose)
    eval_results["avg_logprob"] = sums.mean(axis=1).mean(axis=0)
    eval_results["avg_normlogprob"] = norms.mean(axis=1).mean(axis=0)
    if return_verbose:
        confidences = [
            [confidence_of_completion(x, evaluator.ANSWER_HINT) for x in completions[:num_eval_samples]] for completions in responses
        ]
        avg_conf = np.array(confidences).mean(axis=1).mean(axis=0)
        eval_results["avg_confidence"] = avg_conf

    return eval_results


def register_manual_args(parser):
    parser.add_argument('--manual_prompt_id', type=str, default=None, required=True)
    parser.add_argument('--style_template', type=str, default="default")

def manual_query_result_filename_func(args):
    return "misc/{}--eng{}--{}{}-{}--manual{}--numsamp{}--temp{}--sty{}--predictions.json".format(
        args.task,
        args.engine,
        get_eval_split_abbrev(args),
        args.slice_dev, args.slice_dev + args.num_dev,
        args.manual_prompt_id,
        args.num_samples,
        args.temperature,
        args.style_template
    )

def read_manual_prompt(task, prompt_id, style_template):    
    prompt_lines = read_jsonline(f'manual_prompts/{task}.jsonline')
    d = dict([(x["id"], x) for x in prompt_lines])
    selected = d[prompt_id]
    assert selected["style_template"] == style_template
    return selected["prompt"]

def predict_framework(args):
    train_data, test_data = load_train_test_set(args)
    task_helper = TaskHelper.from_taskname(args.task, args.style_template)

    base_manual_prompt = read_manual_prompt(args.task, args.manual_prompt_id, args.style_template)
    prompts_to_complete = []    
    for test_ex in test_data:
        test_part = task_helper.prompt_func(test_ex, [])
        
        prompts_to_complete.append(
            [base_manual_prompt + task_helper.get_train_sep() + test_part]
        )

    task_max_tokens = task_helper.get_completion_length()
    task_stop_token = task_helper.get_train_sep()
    cache_filename = manual_query_result_filename_func(args)
    responses = run_completion_tasks_with_cache(args, cache_filename, prompts_to_complete, task_max_tokens, task_stop_token)
    responses = [flatten_nested_list(resps_by_example) for resps_by_example in responses]

    eval_results = run_evaluation(args, test_data, responses)
    print_tabular_results("VOTE"+str(args.num_eval_samples), eval_results)

def eval_framework(args):
    _, test_data = load_train_test_set(args)
    responses = read_json(manual_query_result_filename_func(args))
    responses = [flatten_nested_list(resps_by_example) for resps_by_example in responses]
    eval_results = run_evaluation(args, test_data, responses)
    print_tabular_results("VOTE"+str(args.num_eval_samples), eval_results)

def main():
    parser = argparse.ArgumentParser()
    register_base_args(parser)
    register_manual_args(parser)

    args = parser.parse_args()
    assert args.task is not None
    assert args.manual_prompt_id is not None

    config_args_and_api(args)
    if args.run_prediction:
        predict_framework(args)
    else:
        eval_framework(args)

if __name__=="__main__":
    main()
