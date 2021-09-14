from .analytic_free_fb import AnalyticFreeFB
from .free_fb import FreeFB
from .param_sinc_fb import ParamSincFB
from .stft_fb import STFTFB
from .enc_dec import Filterbank, Encoder, Decoder
from .griffin_lim import griffin_lim, misi
from .multiphase_gammatone_fb import MultiphaseGammatoneFB
from .melgram_fb import MelGramFB
import torch


__version__ = "0.3.1"

class MultiParEnc(torch.nn.Module):
    def __init__(self, rescoders, parcoders):
        super(MultiParEnc, self).__init__()
        self.rescoders = rescoders
        self.parcoders = parcoders
        self.stride = rescoders[0].stride
        self.resnum = len(self.rescoders)
        self.n_feats_out = sum([coder.n_feats_out for coder in self.rescoders])

    def forward(self, x):
        assert x.dim()==3
        outputs = [coder(x[:,[0,],:]) for coder in self.rescoders]
        outputs = torch.cat(outputs,1)
        if(x.shape[1]==1):return outputs
        for coderinx in range(len(self.parcoders)):
            outputs = outputs + self.parcoders[coderinx](x[:,[coderinx+1,],:])
        return outputs

def make_multipar_enc_dec(fb_name, n_filters: list,
                 n_channels: int,
                 kernel_size: list, stride: list,
                 weight_list: tuple,
                 who_is_pinv=None, **kwargs):
    fb_class = get(fb_name)
    assert len(n_filters) == len(kernel_size), (n_filters, kernel_size)
    assert len(n_filters) == len(stride), (n_filters, stride)
    enc = torch.nn.ModuleList()
    par = torch.nn.ModuleList()
    dec = torch.nn.ModuleList()
    dec_slice = list()
    for n, k, s in zip(n_filters, kernel_size, stride):
        dec_slice.append(slice((k-kernel_size[0])//2,(kernel_size[0]-k)//2))
        if who_is_pinv in ['dec', 'decoder']:
            fb = fb_class(n, k, stride=s, **kwargs)
            enc.append(Encoder(fb, padding=(k-kernel_size[0])//2))
            # Decoder filterbank is pseudo inverse of encoder filterbank.
            dec.append(Decoder.pinv_of(fb))
        elif who_is_pinv in ['enc', 'encoder']:
            fb = fb_class(n, k, stride=s, **kwargs)
            dec.append(Decoder(fb))
            # Encoder filterbank is pseudo inverse of decoder filterbank.
            enc.append(Encoder.pinv_of(fb))
            enc[-1].padding=(k-kernel_size[0])//2
        else:
            fb = fb_class(n, k, stride=s, **kwargs)
            enc.append(Encoder(fb, padding=(k-kernel_size[0])//2))
            # Filters between encoder and decoder should not be shared.
            fb = fb_class(n, k, stride=s,  **kwargs)
            dec.append(Decoder(fb))
    for n in range(n_channels-1):
        fb = fb_class(sum(n_filters), kernel_size[0], stride=stride[0], **kwargs)
        par.append(Encoder(fb))
    return MultiParEnc(enc, par), MultiresDec(dec,weight_list, dec_slice)


class MultiresEnc(torch.nn.Module):
    def __init__(self, coders):
        super(MultiresEnc, self).__init__()
        self.coders = coders
        self.stride = coders[0].stride
        self.resnum = len(self.coders)
        self.n_feats_out = sum([coder.n_feats_out for coder in self.coders])

    def forward(self, x):
        outputs = [coder(x) for coder in self.coders]
        #n_samples = min([output.shape[-1] for output in outputs])
        #n_samples = [output.shape[-1]-n_samples for output in outputs]
        #outputs = [output[i][..., n_samples[i]:-n_samples[i]] for i in len(outputs)]
        return torch.cat(outputs,1)

class MultiresDec(torch.nn.Module):
    def __init__(self, coders, weight_list, dec_slice):
        super(MultiresDec, self).__init__()
        self.coders = coders
        self.weight_list = weight_list
        self.resnum = list()
        tlen = 0
        for coder in self.coders:
            self.resnum.append(slice(tlen,tlen+coder.filterbank.n_filters))
        self.dec_slice=dec_slice

    def forward(self, x):
        for coderinx in range(len(self.coders)):
            if(coderinx==0):
                output=self.weight_list[coderinx]*self.coders[coderinx](x[:,:,self.resnum[coderinx],:])
            else:
                output+=self.weight_list[coderinx]*self.coders[coderinx](x[:,:,self.resnum[coderinx],:])[:,:,self.dec_slice[coderinx]]
        return output

def make_multiple_enc_dec(fb_name, n_filters: list,
                 kernel_size: list, stride: list,
                 weight_list: tuple,
                 who_is_pinv=None, **kwargs):
    fb_class = get(fb_name)
    assert len(n_filters) == len(kernel_size), (n_filters, kernel_size)
    assert len(n_filters) == len(stride), (n_filters, stride)
    enc = torch.nn.ModuleList()
    dec = torch.nn.ModuleList()
    dec_slice = list()
    for n, k, s in zip(n_filters, kernel_size, stride):
        dec_slice.append(slice((k-kernel_size[0])//2,(kernel_size[0]-k)//2))
        if who_is_pinv in ['dec', 'decoder']:
            fb = fb_class(n, k, stride=s, **kwargs)
            enc.append(Encoder(fb, padding=(k-kernel_size[0])//2))
            # Decoder filterbank is pseudo inverse of encoder filterbank.
            dec.append(Decoder.pinv_of(fb))
        elif who_is_pinv in ['enc', 'encoder']:
            fb = fb_class(n, k, stride=s, **kwargs)
            dec.append(Decoder(fb))
            # Encoder filterbank is pseudo inverse of decoder filterbank.
            enc.append(Encoder.pinv_of(fb))
            enc[-1].padding=(k-kernel_size[0])//2
        else:
            fb = fb_class(n, k, stride=s, **kwargs)
            enc.append(Encoder(fb, padding=(k-kernel_size[0])//2))
            # Filters between encoder and decoder should not be shared.
            fb = fb_class(n, k, stride=s,  **kwargs)
            dec.append(Decoder(fb))
    return MultiresEnc(enc), MultiresDec(dec,weight_list, dec_slice)

class ParEncoder(torch.nn.Module):
    def __init__(self, coders, return_chanwise=False):
        super(ParEncoder, self).__init__()
        self.coders = coders
        self.return_chanwise = return_chanwise
        if(isinstance(coders[0],Encoder)):
            self.n_feats_out = coders[0].n_feats_out
            self.stride = coders[0].stride

    def forward(self, x):
        if(self.return_chanwise):
            output = list()
            for coderinx in range(min(len(self.coders),x.shape[1])):
                output.append(self.coders[coderinx](x[:,[coderinx],:]))
            output = torch.stack(output,1)
            return output
        else:
            output = 0.0
            for coderinx in range(min(len(self.coders),x.shape[1])):
                output = output + self.coders[coderinx](x[:,[coderinx],:])
            return output

class ParDecoder(torch.nn.Module):
    def __init__(self, coders):
        super(ParDecoder, self).__init__()
        self.coders = coders
        if(isinstance(coders[0],Encoder)):
            self.n_feats_out = coders[0].n_feats_out
            self.stride = coders[0].stride

    def forward(self, x):
        output = []
        for coderinx in range(0,min(len(self.coders),x.shape[1])):
            output.append(self.coders[coderinx](x[:,[coderinx],:]))
        return torch.cat(output,1)

def make_parallel_enc_dec(fb_name,
                 n_channels: int,
                 n_filters: int,
                 kernel_size: int, 
                 stride: int,
                 who_is_pinv=None, 
                 use_par_dec=False,
                 return_chanwise=False,
                 **kwargs):
    fb_class = get(fb_name)
    enc = torch.nn.ModuleList()
    dec = torch.nn.ModuleList()
    for n in range(n_channels):
        if who_is_pinv in ['dec', 'decoder']:
            fb = fb_class(n_filters, kernel_size, stride=stride, **kwargs)
            enc.append(Encoder(fb))
            # Decoder filterbank is pseudo inverse of encoder filterbank.
            dec.append(Decoder.pinv_of(fb))
        elif who_is_pinv in ['enc', 'encoder']:
            fb = fb_class(n_filters, kernel_size, stride=stride, **kwargs)
            dec.append(Decoder(fb))
            # Encoder filterbank is pseudo inverse of decoder filterbank.
            enc.append(Encoder.pinv_of(fb))
        else:
            fb = fb_class(n_filters, kernel_size, stride=stride, **kwargs)
            enc.append(Encoder(fb))
            # Filters between encoder and decoder should not be shared.
            fb = fb_class(n_filters, kernel_size, stride=stride, **kwargs)
            dec.append(Decoder(fb))
    if use_par_dec:
        return ParEncoder(enc, return_chanwise=return_chanwise), ParDecoder(dec)
    else:
        return ParEncoder(enc, return_chanwise=return_chanwise), dec[0]

class AttEncoder(torch.nn.Module):
    def __init__(self, coders):
        super(AttEncoder, self).__init__()
        self.coders = coders
        if(isinstance(coders[0],Encoder)):
            self.n_feats_out = coders[0].n_feats_out//3
            self.stride = coders[0].stride

    def forward(self, x):
        output = []
        for coderinx in range(min(len(self.coders),x.shape[1])):
            output.append(self.coders[coderinx](x[:,[coderinx],:]))
        output = torch.stack(output,1) # B C F T
        k_output, q_output, v_output = output.chunk(3,-2)
        w = (k_output.unsqueeze(1) * q_output.unsqueeze(2)).sum(-2, keepdim=True).softmax(2) # B C C 1 T
        output = ((w * v_output.unsqueeze(1)).sum(2) + v_output)/2.0 # B C F T
        return output

def make_attention_enc_dec(fb_name,
                 n_channels: int,
                 n_filters: int,
                 kernel_size: int, 
                 stride: int,
                 who_is_pinv=None, 
                 use_par_dec=False,
                 **kwargs):
    fb_class = get(fb_name)
    enc = torch.nn.ModuleList()
    dec = torch.nn.ModuleList()
    for n in range(n_channels):
        enc_fb = fb_class(n_filters*3, kernel_size, stride=stride, **kwargs)
        enc.append(Encoder(enc_fb))
        # Filters between encoder and decoder should not be shared.
        dec_fb = fb_class(n_filters, kernel_size, stride=stride, **kwargs)
        dec.append(Decoder(dec_fb))
    if use_par_dec:
        return AttEncoder(enc), ParDecoder(dec)
    else:
        return AttEncoder(enc), dec[0]

def make_enc_dec(
    fb_name,
    n_filters,
    kernel_size,
    stride=None,
    sample_rate=8000.0,
    who_is_pinv=None,
    padding=0,
    output_padding=0,
    **kwargs,
    ):
    """Creates congruent encoder and decoder from the same filterbank family.

    Args:
        fb_name (str, className): Filterbank family from which to make encoder
            and decoder. To choose among [``'free'``, ``'analytic_free'``,
            ``'param_sinc'``, ``'stft'``]. Can also be a class defined in a
            submodule in this subpackade (e.g. :class:`~.FreeFB`).
        n_filters (int): Number of filters.
        kernel_size (int): Length of the filters.
        stride (int, optional): Stride of the convolution.
            If None (default), set to ``kernel_size // 2``.
        sample_rate (float): Sample rate of the expected audio.
            Defaults to 8000.0.
        who_is_pinv (str, optional): If `None`, no pseudo-inverse filters will
            be used. If string (among [``'encoder'``, ``'decoder'``]), decides
            which of ``Encoder`` or ``Decoder`` will be the pseudo inverse of
            the other one.
        padding (int): Zero-padding added to both sides of the input.
            Passed to Encoder and Decoder.
        output_padding (int): Additional size added to one side of the output shape.
            Passed to Decoder.
        **kwargs: Arguments which will be passed to the filterbank class
            additionally to the usual `n_filters`, `kernel_size` and `stride`.
            Depends on the filterbank family.
    Returns:
        :class:`.Encoder`, :class:`.Decoder`
    """
    fb_class = get(fb_name)

    if who_is_pinv in ["dec", "decoder"]:
        fb = fb_class(n_filters, kernel_size, stride=stride, sample_rate=sample_rate, **kwargs)
        enc = Encoder(fb, padding=padding)
        # Decoder filterbank is pseudo inverse of encoder filterbank.
        dec = Decoder.pinv_of(fb)
    elif who_is_pinv in ["enc", "encoder"]:
        fb = fb_class(n_filters, kernel_size, stride=stride, sample_rate=sample_rate, **kwargs)
        dec = Decoder(fb, padding=padding, output_padding=output_padding)
        # Encoder filterbank is pseudo inverse of decoder filterbank.
        enc = Encoder.pinv_of(fb)
    else:
        fb = fb_class(n_filters, kernel_size, stride=stride, sample_rate=sample_rate, **kwargs)
        enc = Encoder(fb, padding=padding)
        # Filters between encoder and decoder should not be shared.
        fb = fb_class(n_filters, kernel_size, stride=stride, sample_rate=sample_rate, **kwargs)
        dec = Decoder(fb, padding=padding, output_padding=output_padding)
    return enc, dec


def register_filterbank(custom_fb):
    """Register a custom filterbank, gettable with `filterbanks.get`.

    Args:
        custom_fb: Custom filterbank to register.

    """
    if custom_fb.__name__ in globals().keys() or custom_fb.__name__.lower() in globals().keys():
        raise ValueError(f"Filterbank {custom_fb.__name__} already exists. Choose another name.")
    globals().update({custom_fb.__name__: custom_fb})


def get(identifier):
    """Returns a filterbank class from a string. Returns its input if it
    is callable (already a :class:`.Filterbank` for example).

    Args:
        identifier (str or Callable or None): the filterbank identifier.

    Returns:
        :class:`.Filterbank` or None
    """
    if identifier is None:
        return None
    elif callable(identifier):
        return identifier
    elif isinstance(identifier, str):
        cls = globals().get(identifier)
        if cls is None:
            raise ValueError("Could not interpret filterbank identifier: " + str(identifier))
        return cls
    else:
        raise ValueError("Could not interpret filterbank identifier: " + str(identifier))


# Aliases.
free = FreeFB
analytic_free = AnalyticFreeFB
param_sinc = ParamSincFB
stft = STFTFB
multiphase_gammatone = mpgtf = MultiphaseGammatoneFB

# For the docs
__all__ = [
    "Filterbank",
    "Encoder",
    "Decoder",
    "FreeFB",
    "STFTFB",
    "AnalyticFreeFB",
    "ParamSincFB",
    "MultiphaseGammatoneFB",
    "MelGramFB",
    "griffin_lim",
    "misi",
    "make_enc_dec",
    "make_multipar_enc_dec",
    "make_multiple_enc_dec",
]
