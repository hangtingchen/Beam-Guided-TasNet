#! /bin/bash

set -eu

test_dirs=
nj=5
stage=0
. ./cmd.sh
. ./path.sh
. ${KALDI_ROOT}/egs/wsj/s5/utils/parse_options.sh

green='\033[0;32m'
NC='\033[0m' # No Color
trap 'echo -e "${green}$ $BASH_COMMAND ${NC}"' DEBUG

[ -z "$test_dirs" ] && echo "Empty var test_dirs" && exit 1

if [ $stage -le 0 ];then
    echo "=========Prepare data========="
    for d in $test_dirs;do
        d0=`basename $d`
        [ -d data/spwsj2orig/${d0}_hires ] && rm -r data/spwsj2orig/${d0}_hires
        mkdir -p data/spwsj2orig/${d0}_hires
        python3 ./local_decodespwsj2/transwavpath2.py data/spwsj2orig/${d0}_hires $d 
        utils/fix_data_dir.sh data/spwsj2orig/${d0}_hires
    done
fi

if [ $stage -le 1 ];then
    echo "=========Extract feature========="
    for d in $test_dirs;do
        d0=`basename $d`
        steps/make_mfcc.sh --nj $nj --mfcc-config conf/mfcc_hires.conf \
            --cmd "$train_cmd" data/spwsj2orig/${d0}_hires
        steps/compute_cmvn_stats.sh data/spwsj2orig/${d0}_hires
        utils/fix_data_dir.sh data/spwsj2orig/${d0}_hires
    done
fi


if [ $stage -le 2 ];then
    echo "=========Extract ivector========="
    for d in $test_dirs;do
        d0=`basename $d`
        nspk=$(wc -l <data/spwsj2orig/${d0}_hires/spk2utt)
        steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj "${nspk}" \
            data/spwsj2orig/${d0}_hires exp/sms_single_speaker/nnet3/extractor \
            exp/sms_single_speaker/nnet3/spwsj2/ivectors_${d0}_hires
    done
fi

# ivector-extract-online2 --config=exp/sms_single_speaker/nnet3/ivectors_cv_dev93_hires/conf/ivector_extractor.conf ark:data/sms_single_speaker/cv_dev93_hires/spk2utt scp:data/sms_single_speaker/cv_dev93_hires/feats.scp ark:- | feat-to-len ark:- ark,t:- 

if [ $stage -le 3 ];then
    echo "=========Save ivector========="
    for d in $test_dirs;do
        d0=`basename $d`
        nspk=$(wc -l <data/spwsj2orig/${d0}_hires/spk2utt)
        steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj "${nspk}" \
            data/spwsj2orig/${d0}_hires exp/sms_single_speaker/nnet3/extractor \
            exp/sms_single_speaker/nnet3/spwsj2/ivectors_${d0}_hires
    done
fi

