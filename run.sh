#!/bin/bash
set -e  # Exit on error
[ ! -d asteroid20210105 ] && git clone --branch v0.4.1 https://github.com/asteroid-team/asteroid.git asteroid20210105
[ ! -d asteroid-filterbanks20210105 ] && git clone --branch v0.3.1 https://github.com/asteroid-team/asteroid-filterbanks.git asteroid-filterbanks20210105
cp __init_filterbanks__.py asteroid-filterbanks20210105/asteroid_filterbanks/__init__.py
export PYTHONPATH=`pwd`/asteroid20210105:`pwd`/asteroid-filterbanks20210105
# Main storage directory. You'll need disk space to dump the WHAM mixtures and the wsj0 wav
# files if you start from sphere files.

# Path to the python you'll use for the experiment. Defaults to the current python
# You can run ./utils/prepare_python_env.sh to create a suitable python environment, paste the output here.
python_path=/opt/conda/envs/torch16/bin/python

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
net_num=2 # netnumber=1 means beamtasnet, netnumber=2 means proposed beam-guided tasnet
causal=false
batch_size=12
num_workers=8
lr=0.001
epochs=150

# Evaluation
eval_use_gpu=1
job_num=0
num_job=1
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

if [[ ${net_num} -eq 1 ]];then
  pretrain_epochs=$epochs
elif [[ ${net_num} -eq 2 ]];then
  pretrain_epochs=`expr $epochs / 2`
else
  echo "net_num=1 means beam-tasnet, net_num=2 means beam-guided tasnet. Unexpected net_num=${net_num}" && exit 1;
fi
conf_file=net${net_num}
if $causal;then
  conf_file="${conf_file}_causal.yml"
else
  conf_file="${conf_file}_noncausal.yml"
fi
cp ./local/${conf_file} ./local/conf.yml

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
                --epochs ${pretrain_epochs} \
                --batch_size $batch_size \
                --num_workers $num_workers \
                --exp_dir ${expdir}/ | tee logs/train_${tag}.log
        cp logs/train_${tag}.log $expdir/train.log
fi

if [[ $stage -le 4 ]]; then
  echo "Stage 4: Training"
  mkdir -p logs
  [ ! -d $expdir/precheckpoints ] && mv $expdir/checkpoints $expdir/precheckpoints
  CUDA_VISIBLE_DEVICES=$id $python_path train.py \
                --pretrain False \
                --train_dir $train_dir \
                --valid_dir $valid_dir \
                --task $task \
                --sample_rate $sample_rate \
                --lr $lr \
                --epochs `expr $epochs / 2` \
                --batch_size $batch_size \
                --num_workers $num_workers \
                --exp_dir ${expdir}/ | tee logs/train_${tag}.log
        cp logs/train_${tag}.log $expdir/train2.log
fi

if [[ $stage -le 6 ]]; then
    echo "Stage 6 : Strict check"
    CUDA_VISIBLE_DEVICES=$id $python_path -u eval_strictcheck.py \
                --num_job $num_job --job_num $job_num \
                --task $task \
                --test_dir $test_dir \
                --use_gpu $eval_use_gpu \
                --stage $test_stage \
                --exp_dir ${expdir} | tee logs/eval_${tag}.log
    cp logs/eval_${tag}.log $expdir/eval.log
    exit
fi

