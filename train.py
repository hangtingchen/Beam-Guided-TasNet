import os
import argparse
import json
import ast

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning import seed_everything

from spatial_wsj0_mix import make_dataloaders
from asteroid.losses import PITLossWrapper, pairwise_neg_snr
from model import make_model_and_optimizer, load_best_model
from system import BeamTasNetSystem, BFLoss


# Keys which are not in the conf.yml file can be added here.
# In the hierarchical dictionary created when parsing, the key `key` can be
# found at dic['main_args'][key]

# By default train.py will use all available GPUs. The `id` option in run.sh
# will limit the number of available GPUs for train.py .
seed_everything(seed=0)
parser = argparse.ArgumentParser()
parser.add_argument('--exp_dir', default='exp/tmp',
                    help='Full path to save best validation model')
parser.add_argument('--pretrain', type=ast.literal_eval, required=True,
                    help='whether to pretrain the stage 1 model')

def main(conf):
    # Update number of source values (It depends on the task)
    conf['masknet'].update({'n_src': conf['data']['n_src']})

    # Define model and optimizer
    model, optimizer = make_model_and_optimizer(conf)
    exp_dir = conf['main_args']['exp_dir']
    if(os.path.exists(os.path.join(exp_dir, 'precheckpoints/'))):
        all_ckpt = os.listdir(os.path.join(exp_dir, 'precheckpoints/'))
        all_ckpt=[(ckpt,int("".join(filter(str.isdigit,ckpt)))) for ckpt in all_ckpt]
        all_ckpt.sort(key=lambda x:x[1])
        best_model_path = os.path.join(exp_dir, 'precheckpoints', all_ckpt[-1][0])
        orig=torch.load(best_model_path,map_location='cpu')['state_dict']
        model_statedict = model.state_dict()
        for k in orig.keys():
            model_statedict[k[6:]]=orig[k]
        model.load_state_dict(model_statedict,strict=True)

    train_loader, val_loader = make_dataloaders(**conf['data'],
                                                **conf['training'],
                                                channels=slice(0,4))

    # Define scheduler
    scheduler = None
    if conf['training']['half_lr']:
        scheduler = ReduceLROnPlateau(optimizer=optimizer, factor=0.5,
                                      patience=5)
    # Just after instantiating, save the args. Easy loading in the future.
    os.makedirs(exp_dir, exist_ok=True)
    conf_path = os.path.join(exp_dir, 'conf.yml')
    with open(conf_path, 'w') as outfile:
        yaml.safe_dump(conf, outfile)

    # Define Loss function.
    loss_func = BFLoss()
    system = BeamTasNetSystem(pretrain=conf['main_args']['pretrain'], 
                    model=model, loss_func=loss_func, optimizer=optimizer,
                    train_loader=train_loader, val_loader=val_loader,
                    scheduler=scheduler, config=conf)

    # Define callbacks
    checkpoint_dir = os.path.join(exp_dir, 'checkpoints/')
    checkpoint = ModelCheckpoint(checkpoint_dir, monitor='val_loss',
                                 mode='min', save_top_k=5, verbose=1)
    early_stopping = False
    if conf['training']['early_stop']:
        early_stopping = EarlyStopping(monitor='val_loss', patience=10,
                                       verbose=1)

    # Don't ask GPU if they are not available.
    gpus = -1 if torch.cuda.is_available() else None
    best_model_path = None
    if(os.path.exists(os.path.join(exp_dir, 'checkpoints/'))):
        all_ckpt = os.listdir(os.path.join(exp_dir, 'checkpoints/'))
        all_ckpt=[(ckpt,int("".join(filter(str.isdigit,ckpt)))) for ckpt in all_ckpt if ckpt.find('ckpt')>=0 and ckpt.find('init')<0]
        if(len(all_ckpt)>0):
            all_ckpt.sort(key=lambda x:x[1])
            best_model_path = os.path.join(exp_dir, 'checkpoints', all_ckpt[-1][0])
    print("resume from {}".format(best_model_path))    


    trainer = pl.Trainer(max_epochs=conf['training']['epochs'],
                         checkpoint_callback=checkpoint,
                         resume_from_checkpoint=best_model_path,
                         early_stop_callback=early_stopping,
                         default_save_path=exp_dir,
                         gpus=gpus,
                         distributed_backend='dp',
                         train_percent_check=1.0,  # Useful for fast experiment
                         gradient_clip_val=5.)
    trainer.fit(system)

    best_k = {k: v.item() for k, v in checkpoint.best_k_models.items()}
    with open(os.path.join(exp_dir, "best_k_models.json"), "w") as f:
        json.dump(best_k, f, indent=0)

    # Save best model (next PL version will make this easier)
    best_path = [b for b, v in best_k.items() if v == min(best_k.values())][0]
    state_dict = torch.load(best_path)
    system.load_state_dict(state_dict=state_dict['state_dict'])
    system.cpu()

    to_save = system.model.serialize()
    torch.save(to_save, os.path.join(exp_dir, 'best_model.pth'))


if __name__ == '__main__':
    import yaml
    from pprint import pprint as print
    from asteroid.utils import prepare_parser_from_dict, parse_args_as_dict

    # We start with opening the config file conf.yml as a dictionary from
    # which we can create parsers. Each top level key in the dictionary defined
    # by the YAML file creates a group in the parser.
    with open('local/conf.yml') as f:
        def_conf = yaml.safe_load(f)
    parser = prepare_parser_from_dict(def_conf, parser=parser)
    # Arguments are then parsed into a hierarchical dictionary (instead of
    # flat, as returned by argparse) to facilitate calls to the different
    # asteroid methods (see in main).
    # plain_args is the direct output of parser.parse_args() and contains all
    # the attributes in an non-hierarchical structure. It can be useful to also
    # have it so we included it here but it is not used.
    arg_dic, plain_args = parse_args_as_dict(parser, return_plain_args=True)
    print(arg_dic)
    main(arg_dic)
