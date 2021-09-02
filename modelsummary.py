import os
import random
import soundfile as sf
import torch
import yaml
import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from pprint import pprint

from model import load_avg_model, load_best_model
from torchsummary import summary

parser = argparse.ArgumentParser()
parser.add_argument('--exp_dir', required=True,
                    help='Experiment root')

def main(conf):
    model = load_best_model(conf['train_conf'], conf['exp_dir'])
    summary(model, [(4,4*8000),(2,4,4*8000)],device='cpu')

if __name__ == '__main__':
    args = parser.parse_args()
    arg_dic = dict(vars(args))

    # Load training config
    conf_path = os.path.join(args.exp_dir, 'conf.yml')
    with open(conf_path) as f:
        train_conf = yaml.safe_load(f)
    arg_dic['train_conf'] = train_conf
    main(arg_dic)
