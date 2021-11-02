import json
import os
import torch
from torch import nn
from sklearn.cluster import KMeans

from asteroid import torch_utils
import asteroid_filterbanks as fb
from asteroid.engine.optimizers import make_optimizer
from asteroid_filterbanks.transforms import take_mag, apply_mag_mask
from asteroid.masknn import norms, activations
from asteroid.utils.torch_utils import pad_x_to_y
from mvdr_model import MVDR
from asteroid.losses import PITLossWrapper, pairwise_neg_snr
import numpy as np
import soundfile as sf

EPS = 1e-8

def make_model_and_optimizer(conf):
    enc, dec = fb.make_parallel_enc_dec('free', **conf['filterbank'], use_par_dec=True)
    conf['filterbank']['n_channels']=conf['filterbank']['n_channels']*3
    enc2, dec2 = fb.make_parallel_enc_dec('free', **conf['filterbank'], use_par_dec=True)
    conf['filterbank']['n_channels']=conf['filterbank']['n_channels']//3
    model = Model(enc, dec, enc2, dec2, conf['masknet'])
    # optimizer = make_optimizer(list(model.enc2.parameters())+list(model.dec2.parameters())+list(model.masker2.parameters()), **conf['optim'])
    optimizer = make_optimizer(model.parameters(), **conf['optim'])
    return model, optimizer

class Model(nn.Module):
    def __init__(self, enc, dec, enc2, dec2, net_conf):
        super().__init__()
        self.enc = enc
        self.dec = dec
        self.enc2 = enc2
        self.dec2 = dec2
        self.net_conf = net_conf
        self.masker1 = TDConvNet(
            net_conf["in_chan"], net_conf["n_src"],
            out_chan=self.enc.n_feats_out*net_conf["audio_chan"],
            n_blocks=net_conf["n_blocks"],
            n_repeats=net_conf["n_repeats"],
            bn_chan=net_conf["bn_chan"],
            hid_chan=net_conf["hid_chan"],
            skip_chan=net_conf["skip_chan"],
            conv_kernel_size=net_conf["conv_kernel_size"],
            norm_type=net_conf["norm_type"],
            mask_act=net_conf["mask_act"],
            causal=net_conf['causal'] if net_conf.get('causal') else False,
        )
        self.masker2 = TDConvNet(
            net_conf["in_chan"], net_conf["n_src"],
            out_chan=self.enc2.n_feats_out*net_conf["audio_chan"],
            n_blocks=net_conf["n_blocks"],
            n_repeats=net_conf["n_repeats"],
            bn_chan=net_conf["bn_chan"],
            hid_chan=net_conf["hid_chan"],
            skip_chan=net_conf["skip_chan"],
            conv_kernel_size=net_conf["conv_kernel_size"],
            norm_type=net_conf["norm_type"],
            mask_act=net_conf["mask_act"],
            causal=net_conf['causal'] if net_conf.get('causal') else False,
        )
        self.n_src = net_conf['n_src']
        self.causal = net_conf.get('causal')
        self.mvdr = MVDR(causal=net_conf.get('causal'))
        self.permute = PITLossWrapper(pairwise_neg_snr, pit_from='pw_mtx')
        self.stft_dict = self.mvdr.stft_dict.copy()
        print("Using stft ", self.stft_dict)

    def forward(self, x, s, do_test=False):
        assert int(stage.split(':')[0])>0
        assert int(stage.split(':')[-1])>0
        n_batch, n_src, n_chan, n_samp = s.shape
        if(do_test is False):
            # randperm the channels
            # the fist chan is selected as the ref chan
            inx_slc = torch.randperm(n_chan).to(s.device)
            x = torch.index_select(x,1,inx_slc)
            s = torch.index_select(s,2,inx_slc)
        elif(do_test=='all'):
            pass
            '''
            inx_slc = self.getIdxSet(n_chan,s.device)
            x = torch.cat([torch.index_select(x,1,inx) for inx in inx_slc],0)
            s = torch.cat([torch.index_select(s,2,inx) for inx in inx_slc],0)
            n_batch = len(inx_slc)
            '''

        tf_x = self.enc(x) # b f t
        m = self.masker1(tf_x).view(n_batch, n_src, n_chan, *tf_x.shape[-2:]) # b s c f t
        est_s = torch_utils.pad_x_to_y(self.dec((m * \
            tf_x.unsqueeze(1).unsqueeze(1)).reshape( \
            n_batch*n_src, n_chan, *tf_x.shape[-2:])), \
            s).view(n_batch, n_src, n_chan, -1) # b s c t

        # est_s = self.permute_sig(est_s, causal=self.causal)
        if(stage.split(':')[-1]=='1'):
            assert int(stage.split(":")[0]=='1')
            return est_s, s
        est_bf = self.mvdr(x, self.permute_sig(est_s.detach(), causal=self.causal))[0].detach() # b s c t
        est_bf = est_bf.view(n_batch, n_src*n_chan, n_samp) # b s c t
        est_bf_x  = self.enc2(torch.cat([x, est_bf],1)) # b s*c f t
        m_bf = self.masker2(est_bf_x).view(n_batch, n_src, n_chan, *est_bf_x.shape[-2:]) # b s c f t
        est_s2 = torch_utils.pad_x_to_y(self.dec2((m_bf * est_bf_x.unsqueeze(1).unsqueeze(1)).reshape(n_batch*n_src, n_chan, *est_bf_x.shape[-2:])), s).view(n_batch, n_src, n_chan, -1) # b s c t
        est_bf = est_bf.view(n_batch, n_src, n_chan, n_samp)

        est_bf2 = self.mvdr(x, self.permute_sig(est_s2.detach(), causal=self.causal))[0].detach() # b s c t
        est_bf2 = est_bf2.view(n_batch, n_src*n_chan, n_samp) # b s c t
        est_bf_x  = self.enc2(torch.cat([x, est_bf2],1)) # b s*c f t
        m_bf = self.masker2(est_bf_x).view(n_batch, n_src, n_chan, *est_bf_x.shape[-2:]) # b s c f t
        est_s3 = torch_utils.pad_x_to_y(self.dec2((m_bf * est_bf_x.unsqueeze(1).unsqueeze(1)).reshape(n_batch*n_src, n_chan, *est_bf_x.shape[-2:])), s).view(n_batch, n_src, n_chan, -1) # b s c t
        est_bf2 = est_bf2.view(n_batch, n_src, n_chan, n_samp)

        '''
        est_bf3 = self.mvdr(x, self.permute_sig(est_s3.detach(), causal=self.causal))[0].detach() # b s c t
        est_bf3 = est_bf3.view(n_batch, n_src, n_chan, n_samp) # b s c t
        return est_s, est_bf, est_s2, est_bf2, est_s3, est_bf3, s
        '''

        # return est_s, est_bf, est_s, s
        return est_s, est_bf, est_s2, est_bf2, est_s3, s

    def getIdxSet(self, n_src, device, reverse=False):
        a1=[a0 for a0 in range(n_src)]
        a1.extend([a0 for a0 in range(n_src)])
        a1=torch.tensor(a1).to(device)
        if(reverse):
            return [a1[-n_src-a0:][0:n_src] for a0 in range(n_src)]
        else:
            return [a1[a0:a0+n_src] for a0 in range(n_src)]

    def permute_sig(self, est_sources, causal=False):
        # b s c t
        reest_sources = [est_sources[:,:,0,:],]
        for chan in range(1,est_sources.shape[2]):
            if(causal):
                est_sources_rest = torch.zeros_like(est_sources[:,:,chan,:])
                if(est_sources.shape[-1]<self.stft_dict['kernel_size']):
                    reest_sources.append(self.permute(est_sources[:,:,chan,:], est_sources[:,:,0,:], return_est=True)[1])
                else:
                    est_sources_rest[:,:,0:self.stft_dict['kernel_size']] = self.permute(est_sources[:,:,chan,0:self.stft_dict['kernel_size']], \
                        est_sources[:,:,0,0:self.stft_dict['kernel_size']], return_est=True)[1]
                    for starti in range(self.stft_dict['kernel_size'], est_sources.shape[-1], self.stft_dict['stride']):
                        endi = min(starti+self.stft_dict['stride'],est_sources.shape[-1])
                        est_sources_rest[:,:,starti:endi] = self.permute(est_sources[:,:,chan,0:endi], \
                            est_sources[:,:,0,0:endi], return_est=True)[1][:,:,starti:endi]
                    reest_sources.append(est_sources_rest)
            else:
                reest_sources.append(self.permute(est_sources[:,:,chan,:], est_sources[:,:,0,:], return_est=True)[1])
        return torch.stack(reest_sources,2)

    def strictForward(self, x, do_test=True, stage='1:2'):
        n_chan = x.shape[1]
        bufflen = self.stft_dict['kernel_size'] * 2
        num_padframes=self.stft_dict['kernel_size']//self.stft_dict['stride']-1
        if(self.stft_dict['kernel_size']-self.stft_dict['stride']>0):
            padx = torch.zeros(x.shape[0], x.shape[1], 2*(self.stft_dict['kernel_size']-self.stft_dict['stride']), device=x.device)
        elif(self.stft_dict['kernel_size']-self.stft_dict['stride']==0):
            padx = torch.zeros(x.shape[0], x.shape[1], self.stft_dict['kernel_size'], device=x.device)
        else:
            raise ValueError()
        if(self.net_conf['causal']):
            # b c t
            for starti in range(0, x.shape[-1], self.stft_dict['stride']):
                # frame-by-frame input with additional buffer
                # each frame infer starti -> starti+self.stft_dict['stride'], but will use bufflen for input, and use pasthist for mvdr cal
                # padx = genFlipPadX(x[:,:,0:starti+self.stft_dict['stride']], 2*(self.stft_dict['kernel_size']-self.stft_dict['stride']))
                if(starti <= bufflen):
                    inputx = torch.cat([x[:,:,0:starti+self.stft_dict['stride']],padx],-1)
                    cursg, curbf = self.atomForward(inputx, num_padframes, do_test, stage)
                else:
                    inputx = torch.cat([x[:,:,starti-bufflen:starti+self.stft_dict['stride']], padx],-1)
                    pastsghist = [x[...,0:starti],*[h[...,0:starti] for h in hist[0:len(hist)//2]]]
                    cursg, curbf = self.atomForward(inputx, num_padframes, do_test, stage, pastsghist=pastsghist, bufflen=bufflen)
                curhist = [*cursg, *curbf]
                if(starti==0):
                    hist = curhist
                else:
                    # sf.write('t1mp{}.wav'.format(starti),hist[1].detach().cpu().numpy()[0,0,0,:],8000)
                    # sf.write('t2mp{}.wav'.format(starti),histcur[1].detach().cpu().numpy()[0,0,0,:],8000)
                    if(starti <= bufflen):
                        hist = [torch.cat([h[0],h[1][...,h[0].shape[-1]:]],-1) for h in zip(hist, curhist)]
                    else:
                        hist = [torch.cat([h[0],h[1]],-1) for h in zip(hist, curhist)]
                hist = [h[...,:min(starti+self.stft_dict['stride'],x.shape[-1])] for h in hist]
                # frame by frame permute
                outhist = hist
        else:
            cursg, curbf = self.atomForward(torch.cat([padx, x, padx],-1), num_padframes, do_test, stage)
            curhist = [*cursg, *curbf]
            hist = [h[...,padx.shape[-1]:-padx.shape[-1]] for h in curhist]
            outhist = [h.mean(0,keepdim=True) for h in hist]
        est_sgs = outhist[:len(outhist)//2]
        est_bfs = outhist[len(outhist)//2:]
        return est_sgs, est_bfs

    def atomForward(self, x, num_padframes, do_test=False, stage='1:2', pastsghist=None, bufflen=None):
        assert int(stage.split(':')[0])>0
        assert int(stage.split(':')[-1])>0
        n_batch, n_chan, n_samp = x.shape
        n_src = self.net_conf['n_src']
        tf_x = self.enc(x) # b f t
        m = self.masker1(tf_x).view(n_batch, n_src, n_chan, *tf_x.shape[-2:]) # b s c f t
        est_s = torch_utils.pad_x_to_y(self.dec((m * \
            tf_x.unsqueeze(1).unsqueeze(1)).reshape( \
            n_batch*n_src, n_chan, *tf_x.shape[-2:])), \
            x).view(n_batch, n_src, n_chan, -1) # b s c t
    
        nowestsg = [est_s[...,bufflen:],]
        nowestbf = list()
        for it in range(0,int(stage[0])+1):
            # Here we use est_s[...,bufflen:] to ensure that
            # 1. Only the current frame is actually inferred, and the past information is not modified.
            # 2. The true causal code should only infer things after `bufflen`, that is, the calculation of
            # est_s[...,0:bufflen] is unnecessary, which will reduce the computation cost time if deployed.
            # Besides, we generate the entire est_bf with causal=True. This trick, called "noncausal MVDR 
            # for causal inference", yields the improved signal quality.
            # We also set causal=False for fast inference
            if(it==0):
                est_bf = self.mvdr(torch.cat([pastsghist[0],x[...,bufflen:]],-1) \
                            if isinstance(pastsghist,list) else x, \
                            self.permute_sig( \
                            torch.cat([pastsghist[it+1], est_s.detach()[...,bufflen:]],-1) \
                            if isinstance(pastsghist,list) else est_s.detach(), \
                            causal=False), \
                            causal=False,
                            num_padframes=num_padframes)[0].detach() # b s c t
            else:
                est_bf = self.mvdr(torch.cat([pastsghist[0],x[...,bufflen:]],-1) \
                            if isinstance(pastsghist,list) else x, \
                            self.permute_sig( \
                            torch.cat([pastsghist[it+1], est_s2.detach()[...,bufflen:]], -1) \
                            if isinstance(pastsghist,list) else est_s2.detach(), \
                            causal=False), \
                            causal=False,
                            num_padframes=num_padframes)[0].detach() # b s c t
            if(isinstance(pastsghist,list)):est_bf = est_bf[...,pastsghist[0].shape[-1]-bufflen:]
            nowestbf.append(est_bf[...,bufflen:])
            if(it==int(stage[0])):break
            est_bf = est_bf.view(n_batch, n_src*n_chan, n_samp) # b s c t
            est_bf_x  = self.enc2(torch.cat([x, est_bf],1)) # b s*c f t
            est_bf = est_bf.view(n_batch, n_src, n_chan, n_samp)

            m_bf = self.masker2(est_bf_x).view(n_batch, n_src, n_chan, *est_bf_x.shape[-2:]) # b s c f t
            est_s2 = torch_utils.pad_x_to_y(self.dec2((m_bf * est_bf_x.unsqueeze(1).unsqueeze(1)).reshape(n_batch*n_src, n_chan, *est_bf_x.shape[-2:])), x).view(n_batch, n_src, n_chan, -1) # b s c t
            nowestsg.append(est_s2[...,bufflen:])

        return nowestsg, nowestbf

    def strictOracleForward(self, x, s, do_test=True, stage='1:2'):
        n_chan = x.shape[1]
        if(do_test=='all'):
            inx_slc = self.getIdxSet(n_chan,x.device)
            x = torch.cat([torch.index_select(x,1,inx) for inx in inx_slc],0)
            s = torch.cat([torch.index_select(s,2,inx) for inx in inx_slc],0)
            n_batch = len(inx_slc)
        bufflen = self.stft_dict['kernel_size'] * 2
        num_padframes=self.stft_dict['kernel_size']//self.stft_dict['stride']-1
        if(self.stft_dict['kernel_size']-self.stft_dict['stride']>0):
            padx = torch.zeros(x.shape[0], x.shape[1], 2*(self.stft_dict['kernel_size']-self.stft_dict['stride']), device=x.device)
            pads = torch.zeros(x.shape[0], s.shape[1], x.shape[1], 2*(self.stft_dict['kernel_size']-self.stft_dict['stride']), device=x.device)
        elif(self.stft_dict['kernel_size']-self.stft_dict['stride']==0):
            padx = torch.zeros(x.shape[0], x.shape[1], self.stft_dict['kernel_size'], device=x.device)
            pads = torch.zeros(x.shape[0], s.shape[1], x.shape[1], self.stft_dict['kernel_size'], device=x.device)
        else:
            raise ValueError()
        if(self.net_conf['causal']):
            # b c t
            for starti in range(0, x.shape[-1], self.stft_dict['stride']):
                # frame-by-frame input with additional buffer
                # each frame infer starti -> starti+self.stft_dict['stride'], but will use bufflen for input, and use pasthist for mvdr cal
                # padx = genFlipPadX(x[:,:,0:starti+self.stft_dict['stride']], 2*(self.stft_dict['kernel_size']-self.stft_dict['stride']))
                if(starti <= bufflen):
                    inputx = torch.cat([x[:,:,0:starti+self.stft_dict['stride']],padx],-1)
                    inputs = torch.cat([s[:,:,:,0:starti+self.stft_dict['stride']],pads],-1)
                    cursg, curbf = self.atomOracleForward(inputx, inputs, num_padframes, do_test, stage)
                else:
                    inputx = torch.cat([x[:,:,starti-bufflen:starti+self.stft_dict['stride']], padx],-1)
                    inputs = torch.cat([s[:,:,:,starti-bufflen:starti+self.stft_dict['stride']], pads],-1)
                    pastsghist = [x[...,0:starti],*[h[...,0:starti] for h in hist[0:len(hist)//2]]]
                    cursg, curbf = self.atomOracleForward(inputx, inputs, num_padframes, do_test, stage, pastsghist=pastsghist, bufflen=bufflen)
                curhist = [*cursg, *curbf]
                if(starti==0):
                    hist = curhist
                else:
                    # sf.write('t1mp{}.wav'.format(starti),hist[1].detach().cpu().numpy()[0,0,0,:],8000)
                    # sf.write('t2mp{}.wav'.format(starti),histcur[1].detach().cpu().numpy()[0,0,0,:],8000)
                    if(starti <= bufflen):
                        hist = [torch.cat([h[0],h[1][...,h[0].shape[-1]:]],-1) for h in zip(hist, curhist)]
                    else:
                        hist = [torch.cat([h[0],h[1]],-1) for h in zip(hist, curhist)]
                hist = [h[...,:min(starti+self.stft_dict['stride'],x.shape[-1])] for h in hist]
                # frame by frame permute
                if(do_test=='all'):
                    tmphist = hist.copy()
                    inx_slc = self.getIdxSet(n_chan, x.device, reverse=True)
                    n_samp = hist[-1].shape[-1]
                    for tmphisti in range(len(tmphist)):
                        tmphist[tmphisti] = torch.cat([torch.index_select(tmphist[tmphisti][[inx0]],2,inx) for inx0,inx in enumerate(inx_slc)],0)#b s c t
                        tmphist[tmphisti] = tmphist[tmphisti].permute(1,0,2,3).reshape(1, self.n_src, n_batch*n_chan, n_samp)
                        # causal=False for causal inference, which saves a lot of time
                        tmphist[tmphisti] = self.permute_sig(tmphist[tmphisti], causal=False)
                        tmphist[tmphisti] = tmphist[tmphisti].view(self.n_src, n_batch, n_chan, n_samp).permute(1,0,2,3).contiguous()
                        tmphist[tmphisti] = tmphist[tmphisti].mean(0,keepdim=True)
                    if(starti==0):
                        outhist = tmphist
                    else:
                        outhist = [torch.cat([h[0],h[1][...,h[0].shape[-1]:]],-1) for h in zip(outhist, tmphist)]
                else:
                     outhist = hist
            '''
            # Final permute, this achieves nearly same results
            if(do_test=='all'):
                inx_slc = self.getIdxSet(n_chan, x.device, reverse=True)
                n_samp = hist[-1].shape[-1]
                for histi in range(len(hist)):
                    hist[histi] = torch.cat([torch.index_select(hist[histi][[inx0]],2,inx) for inx0,inx in enumerate(inx_slc)],0) # b s c t
                    hist[histi] = hist[histi].permute(1,0,2,3).reshape(1, self.n_src, n_batch*n_chan, n_samp)
                    hist[histi] = self.permute_sig(hist[histi], causal=self.causal)
                    hist[histi] = hist[histi].view(self.n_src, n_batch, n_chan, n_samp).permute(1,0,2,3).contiguous()
                outhist = [h.mean(0,keepdim=True) for h in hist]
            '''
        else:
            cursg, curbf = self.atomOracleForward(torch.cat([padx, x, padx],-1), torch.cat([pads, s, pads],-1), num_padframes, do_test, stage)
            curhist = [*cursg, *curbf]
            hist = [h[...,padx.shape[-1]:-padx.shape[-1]] for h in curhist]
            if(do_test=='all'):
                inx_slc = self.getIdxSet(n_chan, x.device, reverse=True)
                n_samp = hist[-1].shape[-1]
                for histi in range(len(hist)):
                    hist[histi] = torch.cat([torch.index_select(hist[histi][[inx0]],2,inx) for inx0,inx in enumerate(inx_slc)],0) # b s c t
                    hist[histi] = hist[histi].permute(1,0,2,3).reshape(1, self.n_src, n_batch*n_chan, n_samp)
                    hist[histi] = self.permute_sig(hist[histi], causal=self.causal)
                    hist[histi] = hist[histi].view(self.n_src, n_batch, n_chan, n_samp).permute(1,0,2,3).contiguous()
            outhist = [h.mean(0,keepdim=True) for h in hist]
        est_sgs = outhist[:len(outhist)//2]
        est_bfs = outhist[len(outhist)//2:]
        return est_sgs, est_bfs

    def atomOracleForward(self, x, s, num_padframes, do_test=False, stage='1:2', pastsghist=None, bufflen=None):
        assert int(stage.split(':')[0])>0
        assert int(stage.split(':')[-1])>0
        n_batch, n_chan, n_samp = x.shape
        n_src = self.net_conf['n_src']
        est_s = s # b s c t
    
        nowestsg = [est_s[...,bufflen:],]
        nowestbf = list()
        for it in range(0,int(stage[0])+1):
            # Here we use est_s[...,bufflen:] to ensure that
            # 1. Only the current frame is actually inferred, and the past information is not modified.
            # 2. The true causal code should only infer things after `bufflen`, that is, the calculation of
            # est_s[...,0:bufflen] is unnecessary, which will reduce the computation cost time if deployed.
            # Besides, we generate the entire est_bf with causal=True. This trick, called "noncausal MVDR 
            # for causal inference", yields the improved signal quality.
            # We also set causal=False for fast inference
            if(it==0):
                est_bf = self.mvdr(torch.cat([pastsghist[0],x[...,bufflen:]],-1) \
                            if isinstance(pastsghist,list) else x, \
                            torch.cat([pastsghist[it+1], est_s.detach()[...,bufflen:]],-1) \
                            if isinstance(pastsghist,list) else est_s.detach(), \
                            causal=False,
                            num_padframes=num_padframes)[0].detach() # b s c t
            else:
                est_bf = self.mvdr(torch.cat([pastsghist[0],x[...,bufflen:]],-1) \
                            if isinstance(pastsghist,list) else x, \
                            torch.cat([pastsghist[it+1], est_s2.detach()[...,bufflen:]], -1) \
                            if isinstance(pastsghist,list) else est_s2.detach(), \
                            causal=False,
                            num_padframes=num_padframes)[0].detach() # b s c t
            if(isinstance(pastsghist,list)):est_bf = est_bf[...,pastsghist[0].shape[-1]-bufflen:]
            nowestbf.append(est_bf[...,bufflen:])
            if(it==int(stage[0])):break
            est_bf = est_bf.view(n_batch, n_src, n_chan, n_samp) # b s c t

            est_s2 = est_bf # b s c t
            nowestsg.append(est_s2[...,bufflen:])

        return nowestsg, nowestbf

def load_best_model(train_conf, exp_dir):
    """ Load best model after training.

    Args:
        train_conf (dict): dictionary as expected by `make_model_and_optimizer`
        exp_dir(str): Experiment directory. Expects to find
            `'best_k_models.json'` of `checkpoints` directory in it.

    Returns:
        nn.Module the best (or last) pretrained model according to the val_loss.
    """
    # Create the model from recipe-local function
    model, _ = make_model_and_optimizer(train_conf)
    try:
        # Last best model summary
        with open(os.path.join(exp_dir, 'best_k_models.json'), "r") as f:
            best_k = json.load(f)
        best_model_path = min(best_k, key=best_k.get)
    except FileNotFoundError:
        # Get last checkpoint
        all_ckpt = os.listdir(os.path.join(exp_dir, 'checkpoints/'))
        all_ckpt=[(ckpt,int("".join(filter(str.isdigit,ckpt)))) for ckpt in all_ckpt]
        all_ckpt.sort(key=lambda x:x[1])
        best_model_path = os.path.join(exp_dir, 'checkpoints', all_ckpt[-1][0])
    # Load checkpoint
    checkpoint = torch.load(best_model_path, map_location='cpu')
    # Load state_dict into model.
    model = torch_utils.load_state_dict_in(checkpoint['state_dict'], model)
    model.eval()
    return model

def load_avg_model(train_conf, exp_dir):
    """ Load best model after training.

    Args:
        train_conf (dict): dictionary as expected by `make_model_and_optimizer`
        exp_dir(str): Experiment directory. Expects to find
            `'best_k_models.json'` of `checkpoints` directory in it.

    Returns:
        nn.Module the best (or last) pretrained model according to the val_loss.
    """
    # Create the model from recipe-local function
    model, _ = make_model_and_optimizer(train_conf)
    all_ckpt = os.listdir(os.path.join(exp_dir, 'checkpoints/'))
    all_ckpt=[(ckpt,int("".join(filter(str.isdigit,ckpt)))) for ckpt in all_ckpt if ckpt.find('ckpt')>=0]
    all_ckpt.sort(key=lambda x:x[1])
    best_model_path = [os.path.join(exp_dir, 'checkpoints', ckpt[0]) for ckpt in all_ckpt][-5:]
    # Load checkpoint
    checkpoint = torch.load(best_model_path[0], map_location='cpu')['state_dict']
    print('orig model : {}'.format(best_model_path[0]))
    for i in range(1,len(best_model_path)):
        tmp_ckpt = torch.load(best_model_path[i], map_location='cpu')['state_dict']
        for k in checkpoint.keys():
            checkpoint[k] += tmp_ckpt[k]
        print('avg model : {}'.format(best_model_path[i]))
    for k in list(checkpoint.keys()):
        '''
        if('stft_model' in k):del checkpoint[k]
        else:checkpoint[k] /= float(len(best_model_path))
        '''
        checkpoint[k] /= float(len(best_model_path))
    # Load state_dict into model.
    model = torch_utils.load_state_dict_in(checkpoint, model)
    model.eval()
    return model

class TDConvNet(nn.Module):
    def __init__(self, in_chan, n_src, out_chan=None, n_blocks=8, n_repeats=3,
                 bn_chan=128, bn_chan2=128, hid_chan=512, skip_chan=128, conv_kernel_size=3,
                 norm_type="gLN", mask_act='relu', kernel_size=None, causal=False):
        super(TDConvNet, self).__init__()
        self.in_chan = in_chan
        self.n_src = n_src
        out_chan = out_chan if out_chan else in_chan
        self.out_chan = out_chan
        self.n_blocks = n_blocks
        self.n_repeats = n_repeats
        self.bn_chan = bn_chan
        self.hid_chan = hid_chan
        self.skip_chan = skip_chan
        if kernel_size is not None:
            conv_kernel_size = kernel_size
        self.conv_kernel_size = conv_kernel_size
        self.norm_type = norm_type
        self.mask_act = mask_act
        self.causal = causal
        if causal:
            assert norm_type!='gLN'

        layer_norm = norms.get(norm_type)(in_chan)
        bottleneck_conv = nn.Conv1d(in_chan, bn_chan, 1)
        self.bottleneck = nn.Sequential(layer_norm, bottleneck_conv)
        # Succession of Conv1DBlock with exponentially increasing dilation.
        self.TCN = nn.ModuleList()
        for r in range(n_repeats):
            for x in range(n_blocks):
                padding = (conv_kernel_size - 1) * 2**x if causal else (conv_kernel_size - 1) * 2**x // 2
                self.TCN.append(Conv1DBlock(bn_chan, hid_chan, skip_chan,
                                conv_kernel_size, padding=padding,
                                dilation=2**x, norm_type=norm_type, causal=causal))
        mask_conv_inp = skip_chan if skip_chan else bn_chan
        mask_conv = nn.Conv1d(mask_conv_inp, n_src*out_chan, 1)
        self.mask_net = nn.Sequential(nn.PReLU(), mask_conv)
        # Get activation function.
        mask_nl_class = activations.get(mask_act)
        # For softmax, feed the source dimension.
        self.output_act = mask_nl_class()

    def forward(self, mixture_w):
        batch, n_filters, n_frames = mixture_w.size()
        output = self.bottleneck(mixture_w)
        skip_connection = 0.
        for i in range(len(self.TCN)):
            # Common to w. skip and w.o skip architectures
            tcn_out = self.TCN[i](output)
            if self.skip_chan:
                residual, skip = tcn_out
                skip_connection = skip_connection + skip
            else:
                residual = tcn_out
            output = output + residual
        # Use residual output when no skip connection
        mask_inp = skip_connection if self.skip_chan else output
        score = self.mask_net(mask_inp)
        score = score.view(batch, self.n_src, self.out_chan, n_frames)
        est_mask = self.output_act(score)
        return est_mask

class Conv1DBlock(nn.Module):
    """One dimensional convolutional block, as proposed in [1].

    Args:
        in_chan (int): Number of input channels.
        hid_chan (int): Number of hidden channels in the depth-wise
            convolution.
        skip_out_chan (int): Number of channels in the skip convolution.
            If 0 or None, `Conv1DBlock` won't have any skip connections.
            Corresponds to the the block in v1 or the paper. The `forward`
            return res instead of [res, skip] in this case.
        kernel_size (int): Size of the depth-wise convolutional kernel.
        padding (int): Padding of the depth-wise convolution.
        dilation (int): Dilation of the depth-wise convolution.
        norm_type (str, optional): Type of normalization to use. To choose from

            -  ``'gLN'``: global Layernorm
            -  ``'cLN'``: channelwise Layernorm
            -  ``'cgLN'``: cumulative global Layernorm

    References:
        [1] : "Conv-TasNet: Surpassing ideal time-frequency magnitude masking
        for speech separation" TASLP 2019 Yi Luo, Nima Mesgarani
        https://arxiv.org/abs/1809.07454
    """
    def __init__(self, in_chan, hid_chan, skip_out_chan, kernel_size, padding,
                 dilation, norm_type="gLN", res_chan=None, causal=False):
        super(Conv1DBlock, self).__init__()
        self.skip_out_chan = skip_out_chan
        res_chan = in_chan if res_chan is None else res_chan
        conv_norm = norms.get(norm_type)
        in_conv1d = nn.Conv1d(in_chan, hid_chan, 1)
        depth_conv1d = nn.Conv1d(hid_chan, hid_chan, kernel_size,
                                 padding=padding, dilation=dilation,
                                 groups=hid_chan)
        if causal:
            chomp = Chomp1d(padding)
            self.shared_block = nn.Sequential(in_conv1d, nn.PReLU(),
                                          conv_norm(hid_chan), depth_conv1d,
                                          chomp, nn.PReLU(), conv_norm(hid_chan))
        else:
            self.shared_block = nn.Sequential(in_conv1d, nn.PReLU(),
                                          conv_norm(hid_chan), depth_conv1d,
                                          nn.PReLU(), conv_norm(hid_chan))
        self.res_conv = nn.Conv1d(hid_chan, res_chan, 1)
        if skip_out_chan:
            self.skip_conv = nn.Conv1d(hid_chan, skip_out_chan, 1)

    def forward(self, x):
        """ Input shape [batch, feats, seq]"""
        shared_out = self.shared_block(x)
        res_out = self.res_conv(shared_out)
        if not self.skip_out_chan:
            return res_out
        skip_out = self.skip_conv(shared_out)
        return res_out, skip_out

class Chomp1d(nn.Module):
    """To ensure the output length is the same as the input.
    """
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        """
        Args:
            x: [M, H, Kpad]
        Returns:
            [M, H, K]
        """
        return x[:, :, :-self.chomp_size].contiguous()
