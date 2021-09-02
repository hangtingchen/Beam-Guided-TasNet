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
from torch_complex.tensor import ComplexTensor
from torch_complex import functional as FC
from distutils.version import LooseVersion

is_torch_1_1_plus = LooseVersion(torch.__version__) >= LooseVersion("1.1.0")
EPS = 1e-8
noncausal_stft_dict={
    'n_filters': 4096,
    'kernel_size': 4096,
    'stride':1024,
}

causal_stft_dict={
    'n_filters': 256,
    'kernel_size': 256,
    'stride':256,
}

class STFT(nn.Module):
    def __init__(self, causal):
        super().__init__()
        if(causal):
            self.stft_dict = causal_stft_dict
        else:
            self.stft_dict = noncausal_stft_dict
        enc, dec = fb.make_enc_dec('stft', **self.stft_dict)
        self.enc = enc
        self.dec = dec

    def stft(self,x):
        # x should be  ... , t
        tf = self.enc(x.contiguous())
        # ..., F, T
        return tf

    def istft(self,x,y=None):
        # x ...,f,t
        x=self.dec(x)
        if(y is not None):
            x=torch_utils.pad_x_to_y(x,y)
        return x

def get_causal_power_spectral_density_matrix(observation, normalize=False, causal=False, num_padframes=0):
    '''
    psd = np.einsum('...dft,...eft->...deft', observation, observation.conj()) # (..., sensors, sensors, freq, frames)
    if normalize:
        psd = np.cumsum(psd, axis=-1)/np.arange(1,psd.shape[-1]+1,dtype=np.complex64)
    if(psd.shape[-1]%causal_step==0):
        return psd[...,causal_step-1::causal_step]
    else:
        return np.concatenate([psd[...,causal_step-1::causal_step], psd[...,[-1]]],-1)
    '''
    obsr, obsi = observation.chunk(2,-2) # S C F T
    psdr = torch.einsum('saft,sbft->sabft',obsr,obsr) + torch.einsum('saft,sbft->sabft',obsi,obsi)
    psdi = -torch.einsum('saft,sbft->sabft',obsr,obsi) + torch.einsum('saft,sbft->sbaft',obsr,obsi)
    if(num_padframes>0):
        psdr = psdr[:,:,:,:,:-num_padframes]
        psdi = psdi[:,:,:,:,:-num_padframes]
    if causal:
        psd = torch.cat([psdr,psdi],-2).cumsum(-1) # S C C F T
        if(normalize):
            psd = psd/torch.arange(1,psd.shape[-1]+1,1,dtype=psd.dtype, device=psd.device)[None,None,None,None,:]
        if(num_padframes>0):
            psd = torch.nn.functional.pad(psd,[0, num_padframes, 0, 0, 0, 0],'replicate')
    else:
        psd = torch.cat([psdr,psdi],-2).sum(-1,keepdim=True) # S C C F 1
        if(normalize):
            pad = psd/psd.shape[-1]
    return psd

def get_mvdr_vector(
    psd_s: ComplexTensor,
    psd_n: ComplexTensor,
    reference_vector = 0,
    use_torch_solver: bool = True,
    diagonal_loading: bool = True,
    diag_eps: float = 1e-7,
    eps: float = 1e-8,
) -> ComplexTensor:
    """Return the MVDR (Minimum Variance Distortionless Response) vector:

        h = (Npsd^-1 @ Spsd) / (Tr(Npsd^-1 @ Spsd)) @ u

    Reference:
        On optimal frequency-domain multichannel linear filtering
        for noise reduction; M. Souden et al., 2010;
        https://ieeexplore.ieee.org/document/5089420

    Args:
        psd_s (ComplexTensor): speech covariance matrix (..., F, C, C)
        psd_n (ComplexTensor): observation/noise covariance matrix (..., F, C, C)
        reference_vector (torch.Tensor): (..., C)
        use_torch_solver (bool): Whether to use `solve` instead of `inverse`
        diagonal_loading (bool): Whether to add a tiny term to the diagonal of psd_n
        diag_eps (float):
        eps (float):
    Returns:
        beamform_vector (ComplexTensor): (..., F, C)
    """  # noqa: D400
    if diagonal_loading:
        psd_n = tik_reg(psd_n, reg=diag_eps, eps=eps)

    if use_torch_solver and is_torch_1_1_plus:
        # torch.solve is required, which is only available after pytorch 1.1.0+
        numerator = FC.solve(psd_s, psd_n)[0]
    else:
        numerator = FC.matmul(psd_n.inverse2(), psd_s)
    # ws: (..., C, C) / (...,) -> (..., C, C)
    ws = numerator / (FC.trace(numerator)[..., None, None] + eps)
    # h: (..., F, C_1, C_2) x (..., C_2) -> (..., F, C_1)
    beamform_vector = ws
    return beamform_vector

def tik_reg(mat: ComplexTensor, reg: float = 1e-8, eps: float = 1e-8) -> ComplexTensor:
    """Perform Tikhonov regularization (only modifying real part).

    Args:
        mat (ComplexTensor): input matrix (..., C, C)
        reg (float): regularization factor
        eps (float)
    Returns:
        ret (ComplexTensor): regularized matrix (..., C, C)
    """
    # Add eps
    C = mat.size(-1)
    eye = torch.eye(C, dtype=mat.dtype, device=mat.device)
    shape = [1 for _ in range(mat.dim() - 2)] + [C, C]
    eye = eye.view(*shape).repeat(*mat.shape[:-2], 1, 1)
    with torch.no_grad():
        epsilon = FC.trace(mat).real[..., None, None] * reg
        # in case that correlation_matrix is all-zero
        epsilon = epsilon + eps
    mat = mat + epsilon * eye
    return mat

def make_model_and_optimizer(conf):
    model = Model(4, stft_dict['n_filters']+2)
    model = model.eval()
    return model, None

class MVDR(nn.Module):
    def __init__(self, causal):
        super().__init__()
        self.stft_model = STFT(causal)
        self.causal = causal
        self.stft_dict = self.stft_model.stft_dict.copy()
        print("Torch MVDR causality: {}".format(self.causal))

    def forward(self, x, s, causal=None, num_padframes=0):
        if(causal is None):
            causal = self.causal
        else:
            assert isinstance(causal, bool)
        n_batch, n_src, n_chan, n_samp = s.shape
        x = x.unsqueeze(1).repeat(1,n_src,1,1).view(n_batch*n_src, n_chan, n_samp)
        s = s.view(n_batch*n_src, n_chan, n_samp)

        X = self.stft_model.stft(x) # B*S C F T
        S = self.stft_model.stft(s) # B*S C F T
        N = X - S 
        n_freq, n_frame = S.shape[-2:]
        # print('N ', N.shape)
        Sscm = get_causal_power_spectral_density_matrix(S, normalize=True, causal=causal, num_padframes=num_padframes) # B*S C C F T 
        Nscm = get_causal_power_spectral_density_matrix(N, normalize=True, causal=causal, num_padframes=num_padframes) # B*S C C F T
        # print('N maxtrix ', N.shape)
        Sscm = ComplexTensor(*Sscm.chunk(2,-2)).permute(0,4,3,1,2) # B*S T F C C
        Nscm = ComplexTensor(*Nscm.chunk(2,-2)).permute(0,4,3,1,2)


        est_filt = get_mvdr_vector(Sscm, Nscm) # B*S T F C C
        est_filt = torch.cat([est_filt.real,est_filt.imag],2) # B*S T F C C

        est_filt = est_filt.permute(0,3,4,2,1) # B*S C C F T
        # print('est_filt ', est_filt.shape)
        est_S = self.apply_bf(est_filt,X) # B*S C F T
        est_s = self.stft_model.istft(est_S)
        est_s = torch_utils.pad_x_to_y(est_s, s).view(n_batch, n_src, n_chan, -1) # b*s c t
        s = s.view(n_batch, n_src, n_chan, -1)
        
        return est_s, s

    def apply_bf(self,f,X):
        '''
            f B C C F T
            X B C F T
        '''
        X_real, X_imag = X.unsqueeze(2).chunk(2,-2) # B C 1 F T
        f_real, f_imag = f.chunk(2,-2)
        f_imag = -1.0 * f_imag
        # enhX_real = (X_real * (f_real + torch.ones_like(f_real))).sum(1) - (X_imag * f_imag).sum(1) # B C F T
        # enhX_imag = (X_real * f_imag).sum(1) + (X_imag * (f_real + torch.ones_like(f_real))).sum(1)
        enhX_real = (X_real * f_real).sum(1) - (X_imag * f_imag).sum(1) # B C F T
        enhX_imag = (X_real * f_imag).sum(1) + (X_imag * f_real).sum(1)
        enhX = torch.cat([enhX_real, enhX_imag],2)
        return enhX

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
    best_model_path = [os.path.join(exp_dir, 'checkpoints', ckpt[0]) for ckpt in all_ckpt]
    # Load checkpoint
    checkpoint = torch.load(best_model_path[0], map_location='cpu')['state_dict']
    print('orig model : {}'.format(best_model_path[0]))
    for i in range(1,len(best_model_path)):
        tmp_ckpt = torch.load(best_model_path[i], map_location='cpu')['state_dict']
        for k in checkpoint.keys():
            checkpoint[k] += tmp_ckpt[k]
        print('avg model : {}'.format(best_model_path[i]))
    for k in checkpoint.keys():
        checkpoint[k] /= float(len(best_model_path))
    # Load state_dict into model.
    model = torch_utils.load_state_dict_in(checkpoint, model)
    model.eval()
    return model


