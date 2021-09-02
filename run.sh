#!/bin/bash
set -e  # Exit on error
export PYTHONPATH=/Storage/ASR/chenhangting/Projects/asteroid20210105:/Storage/ASR/chenhangting/Projects/asteroid-filterbanks20210105
# Main storage directory. You'll need disk space to dump the WHAM mixtures and the wsj0 wav
# files if you start from sphere files.
storage_dir=

# If you start from the sphere files, specify the path to the directory and start from stage 0
sphere_dir=  # Directory containing sphere files
# If you already have wsj0 wav files, specify the path to the directory here and start from stage 1
wsj0_wav_dir=
# If you already have the WHAM mixtures, specify the path to the directory here and start from stage 2
wham_wav_dir=
# After running the recipe a first time, you can run it from stage 3 directly to train new models.

# Path to the python you'll use for the experiment. Defaults to the current python
# You can run ./utils/prepare_python_env.sh to create a suitable python environment, paste the output here.
python_path=/opt/conda/envs/torch16/bin/python
# python_path=/home/chenhangting/miniconda3/envs/condacuda10/bin/python

# Example usage
# ./run.sh --stage 3 --tag my_tag --task sep_noisy --id 0,1

# General
stage=0  # Controls from which stage to start
tag=""  # Controls the directory name associated to the experiment
# You can ask for several GPUs using id (passed to CUDA_VISIBLE_DEVICES)
id=$CUDA_VISIBLE_DEVICES
test_stage=1:2

# Data
task=reverb2reverb
sample_rate=8000
mode=min
n_src=2

# Training
batch_size=10
num_workers=8
#optimizer=adam
lr=0.001
epochs=200

# Evaluation
eval_use_gpu=1
num_task=1
task_num=0
test_dir=data/${n_src}speakers/wav8k/max/tt

. utils/parse_options.sh

sr_string=$(($sample_rate/1000))
suffix=${n_src}speakers/wav${sr_string}k/$mode
dumpdir=data/$suffix  # directory to put generated json file

train_dir=$dumpdir/tr
valid_dir=$dumpdir/cv


# Generate a random ID for the run if no tag is specified
uuid=$($python_path -c 'import uuid, sys; print(str(uuid.uuid4())[:8])')
if [[ -z ${tag} ]]; then
        tag=${task}_${sr_string}k${mode}_${uuid}
fi
expdir=exp/train_convtasnet_${tag}
mkdir -p $expdir && echo $uuid >> $expdir/run_uuid.txt
echo "Results from the following experiment will be stored in $expdir"

if [[ $stage -le 3 ]]; then
  echo "Stage 3: Training"
  mkdir -p logs
  CUDA_VISIBLE_DEVICES=$id $python_path train.py \
                --pretrain True \
                --train_dir $train_dir \
                --valid_dir $valid_dir \
                --task $task \
                --sample_rate $sample_rate \
                --lr $lr \
                --epochs $epochs \
                --batch_size $batch_size \
                --num_workers $num_workers \
                --exp_dir ${expdir}/ | tee logs/train_${tag}.log
        cp logs/train_${tag}.log $expdir/train.log
fi

if [[ $stage -le 4 ]]; then
  echo "Stage 4: Training"
  mkdir -p logs
  CUDA_VISIBLE_DEVICES=$id $python_path train.py \
                --pretrain False \
                --train_dir $train_dir \
                --valid_dir $valid_dir \
                --task $task \
                --sample_rate $sample_rate \
                --lr $lr \
                --epochs $epochs \
                --batch_size $batch_size \
                --num_workers $num_workers \
                --exp_dir ${expdir}/ | tee logs/train_${tag}.log
        cp logs/train_${tag}.log $expdir/train2.log
fi


if [[ $stage -le 5 ]]; then
    echo "Stage 5 : Evaluation"
    CUDA_VISIBLE_DEVICES=$id $python_path -u eval.py \
                --task $task \
                --test_dir $test_dir \
                --use_gpu $eval_use_gpu \
                --stage $test_stage \
                --exp_dir ${expdir} | tee logs/eval_${tag}.log
    cp logs/eval_${tag}.log $expdir/eval.log
    exit
fi

if [[ $stage -le 6 ]]; then
    echo "Stage 6 : Strict check"
    CUDA_VISIBLE_DEVICES=$id $python_path -u eval_strictcheck.py \
                --task $task \
                --test_dir $test_dir \
                --use_gpu $eval_use_gpu \
                --stage $test_stage \
                --exp_dir ${expdir} | tee logs/eval_${tag}.log
    cp logs/eval_${tag}.log $expdir/eval.log
    exit
fi

