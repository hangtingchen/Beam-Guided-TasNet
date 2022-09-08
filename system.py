import torch
from torch import nn
import random

from asteroid.engine.system import System
from asteroid.losses import pairwise_neg_snr, PITLossWrapper
from asteroid.filterbanks.transforms import take_mag
from asteroid.filterbanks.enc_dec import Encoder
from asteroid.filterbanks import STFTFB

EPS = 1e-8

class BeamTasNetSystem(System):
    def __init__(self, pretrain, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pretrain = pretrain

    def common_step(self, batch, batch_nb, train=False):
        inputs, targets = self.unpack_data(batch)
        if(self.pretrain):
            est_sig1, sig = self(inputs, targets, do_test=not train, pretrain=True)
        else:
            est_sig1, est_bf, est_sig2, est_bf2, est_sig3, sig = self(inputs, targets, do_test=not train, pretrain=False)
        if(self.pretrain):
            loss, loss_dic = self.loss_func(est_sig1, sig)
        else:
            loss, loss_dic = self.loss_func(est_sig1, est_bf, est_sig2, est_bf2, est_sig3, sig)
        return loss, loss_dic

    def training_step(self, batch, batch_nb):
        loss, loss_dic = self.common_step(batch, batch_nb, train=True)
        tensorboard_logs = loss_dic
        return {'loss': loss, 'log': tensorboard_logs,
                'progress_bar': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        loss, loss_dic = self.common_step(batch, batch_nb, train=False)
        tensorboard_logs = loss_dic
        return {'val_loss': loss, 'log': tensorboard_logs}

    def validation_end(self, outputs):
        loss_names = outputs[0]['log'].keys()
        # Not so pretty for now but it helps.
        tensorboard_logs = {k: \
            torch.stack([x['log'][k] for x in outputs]).mean() \
            for k in loss_names}
        val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        return {'val_loss': val_loss, 'log': tensorboard_logs,
                'progress_bar': tensorboard_logs}

    def validation_epoch_end(self, outputs):
        return self.validation_end(outputs)

    def test_step(self, batch, batch_nb):
        return self.validation_step(batch, batch_nb)

    def test_epoch_end(self, outputs):
        return self.validation_end(outputs)

    def unpack_data(self, batch):
        return batch

class BFLoss(nn.Module):
    def __init__(self, factors=None):
        super().__init__()
        # PIT loss
        self.sig_loss = PITLossWrapper(pairwise_neg_snr)
        self.factors = dict(sig_factor=1.0, mag_factor=0.0) if factors is None else factors

    def forward(self, *args):
        if(len(args)==2):
            est_sig1, sig = args
            # est_sig : B S C N
            B,S,C,N = est_sig1.shape
            est_sig1 = est_sig1.permute(0,2,1,3).reshape(B*C,S,N)
            sig = sig.permute(0,2,1,3).reshape(B*C,S,N)
            sig_loss1 = self.sig_loss(est_sig1, sig)
            n_batch, n_src, _ = sig.shape
            loss_dic = dict(
                sig_loss1 = sig_loss1.mean(),
            )
            loss = sig_loss1.mean()
            return loss, loss_dic
        elif(len(args)==6):
            est_sig1, est_bf, est_sig2, est_bf2, est_sig3, sig = args
            # est_sig : B S C N
            B,S,C,N = est_sig1.shape
            est_sig1 = est_sig1.permute(0,2,1,3).reshape(B*C,S,N)
            est_sig2 = est_sig2.permute(0,2,1,3).reshape(B*C,S,N)
            est_sig3 = est_sig3.permute(0,2,1,3).reshape(B*C,S,N)
            sig = sig.permute(0,2,1,3).reshape(B*C,S,N)
            sig_loss1 = self.sig_loss(est_sig1, sig)
            sig_loss2 = self.sig_loss(est_sig2, sig)
            sig_loss3 = self.sig_loss(est_sig3, sig)
            n_batch, n_src, _ = sig.shape
            loss_dic = dict(
                sig_loss1 = sig_loss1.mean(),
                sig_loss2 = sig_loss2.mean(),
                sig_loss3 = sig_loss3.mean(),
            )
            loss = sig_loss1.mean() + sig_loss2.mean()+ sig_loss3.mean()
            return loss, loss_dic
        else:
            raise NotImplementedError()
