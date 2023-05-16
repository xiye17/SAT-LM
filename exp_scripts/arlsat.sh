EVAL=${1:-"test"}

NUM_DEV=${ND:-"-1"}

main_exp()
{
    # cot
    OPENAI_API_KEY=${KEY} python run_manual.py --task arlsat --run_pred --batch_size 5 --num_samples 1 --temperature 0.0 --style_template cot --manual_prompt_id cot --num_dev ${NUM_DEV} --do_impose ${FLAG}
    # satlm
    OPENAI_API_KEY=${KEY} python run_multistage.py --task arlsat --run_pred --batch_size 5 --num_samples 1 --temperature 0.0 --sig_prompt_id sigz3 --trans_setting setupsatlm --eval_split ${EVAL} --num_dev ${NUM_DEV} --do_impose ${FLAG}
}

main_exp
