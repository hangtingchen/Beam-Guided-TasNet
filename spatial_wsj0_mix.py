import torch
from torch.utils import data
import json
import os
import numpy as np
import soundfile as sf
from torch.utils.data import Sampler

SPWSJ0MIX_TASKS = [
    'reverb2clean',
    'reverb2anechoic',
    'reverb2early',
    'reverb2reverb',
    'clean2clean',
]

def make_dataloaders(train_dir, valid_dir, task, n_src=2, sample_rate=8000,
                     segment=4.0, batch_size=4, num_workers=None, channels=slice(0,1),
                     **kwargs):
    num_workers = num_workers if num_workers else batch_size
    train_set = SPWSJ0MIXDataset(train_dir, task=task, n_src=n_src,
                               sample_rate=sample_rate,
                               segment=segment,
                               channels=channels)
    val_set = SPWSJ0MIXDataset(valid_dir, task=task, n_src=n_src,
                             sample_rate=sample_rate,
                             segment=segment,
                             channels=channels)
    train_loader = data.DataLoader(train_set, shuffle=True,
                                   batch_size=batch_size,
                                   num_workers=num_workers,
                                   drop_last=True)
    val_loader = data.DataLoader(val_set, shuffle=False,
                                 batch_size=batch_size,
                                 num_workers=num_workers,
                                 drop_last=True)
    return train_loader, val_loader

def make_multiple_dataloaders(batch_size, dict_2spk, dict_3spk, num_workers=None, **kwargs):
    num_workers = num_workers if num_workers else batch_size
    if(kwargs.get('channels')):channels=kwargs['channels']
    else:channels=slice(0,1)
    train2_db = SPWSJ0MIXDataset(dict_2spk['train_dir'], task=dict_2spk['task'], n_src=dict_2spk['n_src'],
                               sample_rate=dict_2spk['sample_rate'],
                               channels=channels,
                               segment=dict_2spk['segment'])
    valid2_db = SPWSJ0MIXDataset(dict_2spk['valid_dir'], task=dict_2spk['task'], n_src=dict_2spk['n_src'],
                               sample_rate=dict_2spk['sample_rate'],
                               channels=channels,
                               segment=dict_2spk['segment'])
    train3_db = SPWSJ0MIXDataset(dict_3spk['train_dir'], task=dict_3spk['task'], n_src=dict_3spk['n_src'],
                               sample_rate=dict_3spk['sample_rate'],
                               channels=channels,
                               segment=dict_3spk['segment'])
    valid3_db = SPWSJ0MIXDataset(dict_3spk['valid_dir'], task=dict_3spk['task'], n_src=dict_3spk['n_src'],
                               sample_rate=dict_3spk['sample_rate'],
                               channels=channels,
                               segment=dict_3spk['segment'])
    train_db = data.dataset.ConcatDataset([train2_db,train3_db])

    train_loader = data.DataLoader(train_db,
                                   batch_sampler=SameSpeakerSampler(batch_size, train_db.cummulative_sizes),
                                   num_workers=num_workers,)

    val_loader = data.DataLoader(valid3_db, shuffle=False,
                                 batch_size=batch_size,
                                 num_workers=num_workers,
                                 drop_last=True)
    return train_loader, val_loader

class SPWSJ0MIXDataset(data.Dataset):
    dataset_name = 'spatial-wsj0-mix'

    def __init__(self, json_dir, task, n_src=2, sample_rate=8000, segment=4.0, channels=slice(0,1)):
        super().__init__()
        # Task setting
        assert task in SPWSJ0MIX_TASKS, task
        self.task = task.split('2')
        self.json_dir = json_dir
        self.sample_rate = sample_rate
        if segment is None:
            self.seg_len = None
        else:
            self.seg_len = int(segment * sample_rate)
        self.n_src = n_src
        self.channels = channels
        self.like_test = self.seg_len is None
        # Load json files
        mix_json = os.path.join(json_dir, 'mix_{}.json'.format(self.task[0]))
        sources_json = [os.path.join(json_dir, source + self.task[1] + '.json') for
                        source in [f"s{n+1}_" for n in range(n_src)]]
        print("load data from {} and {}".format(mix_json, sources_json))
        with open(mix_json, 'r') as f:
            mix_infos = json.load(f)
        sources_infos = []
        for src_json in sources_json:
            with open(src_json, 'r') as f:
                sources_infos.append(json.load(f))
        # Filter out short utterances only when segment is specified
        orig_len = len(mix_infos)
        drop_utt, drop_len = 0, 0
        if not self.like_test:
            for i in range(len(mix_infos) - 1, -1, -1):  # Go backward
                if mix_infos[i][1] < self.seg_len:
                    drop_utt += 1
                    drop_len += mix_infos[i][1]
                    del mix_infos[i]
                    for src_inf in sources_infos:
                        del src_inf[i]

        print("Drop {} utts({:.2f} h) from {} (shorter than {} samples)".format(
            drop_utt, drop_len/sample_rate/36000, orig_len, self.seg_len))
        self.mix = mix_infos
        self.sources = sources_infos

    @property
    def taskstr(self):
        return "2".join(self.task)

    def __len__(self):
        return len(self.mix)

    def __getitem__(self, idx):
        """ Gets a mixture/sources pair.
        Returns:
            mixture, vstack([source_arrays])
        """
        # Random start
        if self.mix[idx][1] == self.seg_len or self.like_test:
            rand_start = 0
        else:
            rand_start = np.random.randint(0, self.mix[idx][1] - self.seg_len)
        if self.like_test:
            stop = None
        else:
            stop = rand_start + self.seg_len
        # Load mixture
        x, _ = sf.read(self.mix[idx][0], start=rand_start,
                       stop=stop, dtype='float32')
        seg_len = torch.as_tensor([len(x)])
        if(len(x.shape)>1):
            assert len(x.shape)==2, x.shape
            x=x.transpose([1,0])[self.channels,:]
            if(x.shape[0]==1):x=x.squeeze(0)
        # Load sources
        source_arrays = []
        for src in self.sources:
            if src[idx] is None:
                # Target is filled with zeros if n_src > default_nsrc
                s = np.zeros((seg_len, ))
            else:
                s, _ = sf.read(src[idx][0], start=rand_start,
                               stop=stop, dtype='float32')
            if(len(s.shape)>1):
                assert len(s.shape)==2, s.shape
                s=s.transpose([1,0])[self.channels,:]
                if(s.shape[0]==1):s=s.squeeze(0)
            source_arrays.append(s)
        sources = torch.from_numpy(np.stack(source_arrays, axis=0))
        return torch.from_numpy(x), sources

    def get_infos(self):
        """ Get dataset infos (for publishing models).

        Returns:
            dict, dataset infos with keys `dataset`, `task` and `licences`.
        """
        infos = dict()
        infos['dataset'] = self.dataset_name
        infos['task'] = 'sep_clean'
        infos['licenses'] = [wsj0_license]
        return infos

class SameSpeakerSampler(Sampler):
    def __init__(self, batch_size, cum_sizes):
        self.batch_size = batch_size
        self.cum_sizes = cum_sizes
        self.ind_sizes = [(cum_sizes[i]-cum_sizes[i-1])//self.batch_size*self.batch_size for i in range(1,len(cum_sizes))]
        self.ind_sizes = [cum_sizes[0]//self.batch_size*self.batch_size, *self.ind_sizes]

    def __iter__(self):
        for i in range(len(self.cum_sizes)):
            if(i==0):
                indxs = [np.array(range(self.cum_sizes[i])),]
            else:
                indxs.append(np.array(range(self.cum_sizes[i-1],self.cum_sizes[i])))
        for i in range(len(indxs)):
            np.random.shuffle(indxs[i])
            indxs[i] = indxs[i][0:self.ind_sizes[i]].reshape(-1,self.batch_size)
        indxs = np.concatenate(indxs,axis=0)
        assert(indxs.shape[0]==len(self)),(self.ind_sizes.shape,len(self))
        np.random.shuffle(indxs)
        return (indxs[i,:].tolist() for i in range(indxs.shape[0]))

    def __len__(self):
        return np.sum([i//self.batch_size for i in self.ind_sizes])
