# SAT-LM
SATisfiability-Aided Language Models using Declarative Prompting

## Setup
* python==3.8
* requirements: `pip install -r requirements.txt`
* Set OPENAI KEY: `export KEY=yourkey`

## Experiments
Preparation:

`mkdir misc tmp`


#### Experiments on Arithemetic Reasoning

GSM:

`sh exp_scripts/gsm.sh test`

GSM-system:

`sh exp_scripts/gsm.sh system`

Algebra:

`sh exp_scripts/gsm.sh algebra`

#### Experiments on Logical Reasoning
ARLSAT:

`sh exp_scripts/arlsat.sh`

CLUTRR:

`sh exp_scripts/clutrr.sh`

ProofWriter:

`sh exp_scripts/proofd5.sh`

#### Prompts

Prompts used in our experiments are stored as jsonline file in `manual_prompts/`

## Citation

```
@article{satlm,
  title = {SATisfiability-Aided Language Models using Declarative Prompting},
  author = {Xi Ye, Qiaochu Chen, Isil Dillig, and Greg Durrett},
  ournal = {arXiv},
  year = {2023},
}
```