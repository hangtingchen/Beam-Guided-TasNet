#! /bin/bash

# This script is an adjusted version of kaldis wsj run_tdnn_1g.sh script.
# We reduced the number tdnn-f layer, batch size and training jobs to
# fit smaller gpus.
# Training and evaluation on wsj_8k leads to the following WER:
#                             this script  tdnn1g_sp
# WER dev93 (tgpr)               7.40        6.68
# WER eval92 (tgpr)              5.59        4.54
# Training and evaluation on sms_wsj with a single speaker leads to the following WER:
# WER cv_dev93 (tgpr)            12.20
# WER test_eval92 (tgpr)         8.93

# Exit on error: https://stackoverflow.com/a/1379904/911441
set -e

nj=5
dataset=sms_single_speaker
test_dirs=
root_set=tt
use_oracle=

stage=0
nnet3_affix=       # affix for exp dirs, e.g. it was _cleaned in tedlium.
# Options which are not passed through to run_ivector_common.sh
affix=1a   #affix for TDNN+LSTM directory e.g. "1a" or "1b", in case we change the configuration.

# training chunk-options
chunk_width=140,100,160

echo "$0 $@"

. ./cmd.sh
. ./path.sh
. ${KALDI_ROOT}/egs/wsj/s5/utils/parse_options.sh

green='\033[0;32m'
NC='\033[0m' # No Color
trap 'echo -e "${green}$ $BASH_COMMAND ${NC}"' DEBUG

dir=exp/$dataset/chain${nnet3_affix}/tdnn${affix}_sp
tree_dir=exp/$dataset/chain${nnet3_affix}/tree_a_sp
# the 'lang' directory is created by this script.
# If you create such a directory with a non-standard topology
# you should probably name it differently.
lang=data/lang_chain


if [ $stage -le 0 ];then
    echo "=========Prepare data========="
    for d in $test_dirs;do
        d0=`dirname $d | xargs basename`
        [ -d data/spwsj2mix_enh_20200923/${root_set}_${d0}_hires ] && rm -r data/spwsj2mix_enh_20200923/${root_set}_${d0}_hires
        mkdir -p data/spwsj2mix_enh_20200923/${root_set}_${d0}_hires
        python3 ./local_decodespwsj2/transwavpath${use_oracle}.py data/spwsj2mix_enh_20200923/${root_set}_${d0}_hires $d 
        utils/fix_data_dir.sh data/spwsj2mix_enh_20200923/${root_set}_${d0}_hires
    done
fi

if [ $stage -le 1 ];then
    echo "=========Extract feature========="
    for d in $test_dirs;do
        d0=`dirname $d | xargs basename`
        steps/make_mfcc.sh --nj $nj --mfcc-config conf/mfcc_hires.conf \
            --cmd "$train_cmd" data/spwsj2mix_enh_20200923/${root_set}_${d0}_hires
        steps/compute_cmvn_stats.sh data/spwsj2mix_enh_20200923/${root_set}_${d0}_hires
        utils/fix_data_dir.sh data/spwsj2mix_enh_20200923/${root_set}_${d0}_hires
    done
fi


################################################################################
# Decode on the dev set with lm rescoring
#############################################################################
if [ $stage -le 19 ]; then
  # The reason we are using data/lang here, instead of $lang, is just to
  # emphasize that it's not actually important to give mkgraph.sh the
  # lang directory with the matched topology (since it gets the
  # topology file from the model).  So you could give it a different
  # lang directory, one that contained a wordlist and LM of your choice,
  # as long as phones.txt was compatible.

  utils/lang/check_phones_compatible.sh \
    data/lang_test_tgpr/phones.txt $lang/phones.txt
fi
if [ $stage -le 20 ]; then
  frames_per_chunk=$(echo $chunk_width | cut -d, -f1)
  rm $dir/.error 2>/dev/null || true

for d in $test_dirs;do
    d0=`dirname $d | xargs basename`
    data_affix=${root_set}_${d0}
    nspk=$(wc -l < data/spwsj2mix_enh_20200923/${root_set}_${d0}_hires/spk2utt)
    steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj $nj \
      data/spwsj2mix_enh_20200923/${root_set}_${d0}_hires exp/$dataset/nnet3${nnet3_affix}/extractor \
      exp/$dataset/nnet3${nnet3_affix}/ivectors_${root_set}_${d0}_hires
    lmtype=tgpr
    steps/nnet3/decode.sh \
        --acwt 1.0 --post-decode-acwt 10.0 \
        --extra-left-context 0 --extra-right-context 0 \
        --extra-left-context-initial 0 \
        --extra-right-context-final 0 \
        --frames-per-chunk $frames_per_chunk \
        --nj $nj --cmd "$decode_cmd"  --num-threads 4 \
        --online-ivector-dir exp/$dataset/nnet3${nnet3_affix}/ivectors_${root_set}_${d0}_hires \
        $tree_dir/graph_${lmtype} data/spwsj2mix_enh_20200923/${root_set}_${d0}_hires ${dir}/decode_${lmtype}_${data_affix} || exit 1
    cat ${dir}/decode_${lmtype}_${data_affix}/scoring_kaldi/best_wer
  done
fi
