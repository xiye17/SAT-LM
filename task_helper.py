from utils import *
import os

def load_train_test_set(args):
    if os.path.exists("data/{}_{}.json".format(args.task, "train")):
        train_data = read_json("data/{}_{}.json".format(args.task, "train"))
    else:
        train_data = []
    eval_split = args.eval_split
    dev_data = read_json("data/{}_{}.json".format(args.task, eval_split))
    if args.num_train == -1:
        args.num_train = len(train_data)
    if args.num_dev == -1:
        args.num_dev = len(dev_data)
    train_data = train_data[args.slice_train:args.slice_train+args.num_train]
    dev_data = dev_data[args.slice_dev:args.slice_dev+args.num_dev]
    return train_data, dev_data

class TaskHelper:
    style_to_completion_length = {}
    style_to_train_sep = {}

    def __init__(self, style):
        self.style = style

    @classmethod
    def from_taskname(cls, taskname, style):
        if taskname == "gsm":
            return GSMTaskHelper(style)
        elif taskname == "clutrr":
            return CLUTRRTaskHelper(style)
        elif taskname == "proofd5":
            return ProofD5TaskHelper(style)
        elif taskname == "arlsat":
            return ArLSATTaskHelper(style)
        else:
            raise RuntimeError("Not Implemented Yet")

    def prompt_func(self, test_ex, shots):
        raise RuntimeError("Not Implemented Yet")

    def get_completion_length(self):
        return self.style_to_completion_length[self.style]

    def get_train_sep(self):
        return self.style_to_train_sep[self.style]


class GSMTaskHelper(TaskHelper):
    style_to_completion_length = {
        "std": 32,
        "cot": 160,
        "proglm": 320,
        "satlm": 320,
        "satcotsolver": 576,
    }

    style_to_train_sep = {
        "std": "\n\n",
        "cot": "\n\n",
        "proglm": "\n\n\n\n\n\n",
        "satlm": "\n\n\n\n\n\n",
        "satcotsolver": "\n\n\n\n\n\n",
    }

    def prompt_func(self, test_ex, shots):
        if self.style == "std" or self.style == "cot":
            return self.cot_prompt(test_ex, shots)
        elif self.style == "proglm":
            return self.proglm_prompt(test_ex, shots)
        elif self.style == "satlm":
            return self.satlm_prompt(test_ex, shots)
        elif self.style == "satcotsolver":
            return self.satlm_prompt(test_ex, shots)
        else:
            raise RuntimeError("Not Implemented Yet")

    def cot_prompt(self, test_ex, shots):
        assert len(shots) == 0
        test_example = "Q: {}\nA:".format(test_ex["question"])
        return test_example

    def proglm_prompt(self, test_ex, shots):
        assert len(shots) == 0
        test_example = "Q: {}\n\n# solution in Python:\n\n\n".format(test_ex["question"])
        return test_example

    def satlm_prompt(self, test_ex, shots):
        return self.proglm_prompt(test_ex, shots)


class ProofWriterTaskHelper(TaskHelper):
    style_to_completion_length = {
        "std": 16,
        "cot": 512,
        "satlm": 768,
        "proglm": 512,
    }

    style_to_train_sep = {
        "std": "\n\n\n",
        "cot": "\n\n\n",
        "satlm": "\n\n\n\n\n",
        "proglm": "\n\n\n\n\n",
    }

    def prompt_func(self, test_ex, shots):
        if self.style == "cot":
            return self.cot_prompt(test_ex, shots)
        elif self.style == "std":
            return self.std_prompt(test_ex, shots)
        elif self.style == "satlm":
            return self.satlm_prompt(test_ex, shots)
        elif self.style == "proglm":
            return self.satlm_prompt(test_ex, shots)
        else:
            raise RuntimeError("Not Implemented Yet")

    def cot_prompt(self, test_ex, shots):
        assert len(shots) == 0
        test_example = (
            'Here are some facts and rules:\n' + 
            '\n'.join(test_ex["context"]) + 
            'Does it imply that the statement "{}" is True?\n'.format(test_ex["question"].rstrip('.')) +
            'Reasoning:\n'
        )
        return test_example

    def std_prompt(self, test_ex, shots):
        assert len(shots) == 0
        test_example = (
            'Here are some facts and rules:\n' + 
            '\n'.join(test_ex["context"]) + 
            'Does it imply that the statement "{}" is True?\n'.format(test_ex["question"].rstrip('.')) +
            'Answer:'
        )
        return test_example

    def satlm_prompt(self, test_ex, shots):
        assert len(shots) == 0
        
        test_example = (
            '"""\n' +
            'Here are some facts and rules:\n' + 
            '\n'.join(test_ex["context"]) + 
            '\nQuestion: The statement "{}" is True or False?\n'.format(test_ex["question"].rstrip('.')) +
            '"""\n' + 
            '# solution in Python:\n' +
            'def solution():\n'
        )
        return test_example


class ProofD5TaskHelper(ProofWriterTaskHelper):
    pass


class CLUTRRTaskHelper(TaskHelper):
    style_to_completion_length = {
        "proglm": 512,
        "satlm": 512,
        "satcotsolver": 768,
    }

    style_to_train_sep = {
        "proglm": "\n\n",
        "satlm": "\n\n\n\n\n",
        "satcotsolver": "\n\n\n\n\n",
    }

    def prompt_func(self, test_ex, shots):
        if self.style == "proglm":
            return self.proglm_prompt(test_ex, shots)
        elif self.style == "satlm":
            return self.satlm_prompt(test_ex, shots)
        elif self.style == "satcotsolver":
            return self.satlm_prompt(test_ex, shots)
        else:
            raise RuntimeError("Not Implemented Yet")

    def proglm_prompt(self, test_ex, shots):
        assert len(shots) == 0
        test_example = ("# Context: {}\n# Question: How is [{}] related to [{}]?\n"
            + "# To answer this question, we write a program to answer the following subquestions:\n").format(
            test_ex["context"], test_ex["query"][1], test_ex["query"][0]
        )
        return test_example

    def satlm_prompt(self, test_ex, shots):
        assert len(shots) == 0
        test_example = '"""\n{}\nQuestion: How is [{}] related to [{}]?\n"""\n'.format(
            test_ex["context"], test_ex["query"][1], test_ex["query"][0]
        )
        return test_example


class LongContextMCQAHelper(TaskHelper):
    style_to_completion_length = {
        "std": 16,
        "cot": 512,
        "satlm": 1024,
    }

    style_to_train_sep = {
        "std": "\n\n",
        "cot": "\n\n\n\n",
        "satlm": "\n\n\n\n",
    }

    CHOICE_IDX = ["(A)", "(B)", "(C)", "(D)", "(E)"]
    CODE_HEADER = "### write python code to answer the question"
    CODE_BLOCK_COMMENT = '"""'
    def prompt_func(self, test_ex, shots):
        if self.style == "std":
            return self.std_prompt(test_ex, shots)
        elif self.style == "cot":
            return self.cot_prompt(test_ex, shots)
        elif self.style == "satlm":
            return self.satlm_prompt(test_ex, shots)
        else:
            raise RuntimeError("Not Implemented Yet")

    def std_prompt(self, test_ex, shots):
        def _single_ex_func(ex, is_train):
            choice_str = "\n".join([self.CHOICE_IDX[i] + " " + x for (i, x) in enumerate(ex["choices"])])
            p_ex = "{}\nQuestion: {}\nChoices:\n{}\nAnswer:".format(ex["context"], ex["question"], choice_str)
            if is_train:
                p_ex = p_ex + " The answer is {}.".format(self.CHOICE_IDX[ex["label"]])
            return p_ex

        showcase_examples = [
            _single_ex_func(s, True) for s in shots
        ]
        test_example = [_single_ex_func(test_ex, False)]
        return self.get_train_sep().join(showcase_examples + test_example)

    def cot_prompt(self, test_ex, shots):
        def _single_ex_func(ex, is_train):
            assert not is_train
            choice_str = "\n".join([self.CHOICE_IDX[i] + " " + x for (i, x) in enumerate(ex["choices"])])
            p_ex = "{}\nQuestion: {}\nChoices:\n{}\nAnswer:".format(ex["context"], ex["question"], choice_str)
            return p_ex

        showcase_examples = [
            _single_ex_func(s, True) for s in shots
        ]
        test_example = [_single_ex_func(test_ex, False)]
        return  self.get_train_sep().join(showcase_examples + test_example)

    def satlm_prompt(self, test_ex, shots):
        def _single_ex_func(ex, is_train):
            assert not is_train
            choice_str = "\n".join([self.CHOICE_IDX[i] + " " + x for (i, x) in enumerate(ex["choices"])])
            p_ex = "{}\n{}\n{}\nQuestion: {}\nChoices:\n{}\n{}\n".format(
                self.CODE_HEADER,
                self.CODE_BLOCK_COMMENT,
                ex["context"], ex["question"], choice_str,
                self.CODE_BLOCK_COMMENT)
            return p_ex

        showcase_examples = [
            _single_ex_func(s, True) for s in shots
        ]
        test_example = [_single_ex_func(test_ex, False)]
        return  self.get_train_sep().join(showcase_examples + test_example)


class ArLSATTaskHelper(LongContextMCQAHelper):
    pass
