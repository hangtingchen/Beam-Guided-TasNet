# Beam-Guided-TasNet
Beam-Guided TasNet: An Iterative Speech Separation Framework with Multi-Channel Output

Please refer to [Preprint Paper](https://arxiv.org/abs/2102.02998) 

## Contents
- [Beam-Guided-TasNet](#beam-guided-tasnet)
  - [Contents](#contents)
  - [Dataset](#dataset)
  - [Installation](#installation)
  - [Train](#train)
  - [Test](#test)
  - [Results](#results)
    - [Noncausal](#noncausal)
    - [Causal](#causal)
    - [Oracle test](#oracle-test)
  - [Acknowledgement](#acknowledgement)
  - [Citing us](#citing-us)

## Dataset
([↑up to contents](#contents))

We use dataset `spatialize_wsj0-mix`, which can be created following [[link]](https://www.merl.com/demos/deep-clustering)

## Installation
([↑up to contents](#contents))

This recipe requires `asteroid` and `asteroid-filterbanks`. 
The script `run.sh` will automatically download these requirements in the current directory. Some requirement may not be satisfied. The best way is to run this recipe directly and errors will tell the uninstalled modules.

Paths should be modified according to your environment : 
The json dataset
```bash
sed -i 's?/path/to/dataset?YOUR_PATH?g' data/2speakers/wav8k/*/*/*.json
```
The `run.sh` :
```bash
python_path=YOUR_PYTHON_PATH
```

## Train
([↑up to contents](#contents))

we recommend running :
```bash
./run.sh --id $id 
```

## Test
([↑up to contents](#contents))

You can give a strict check by running :
```bash
./run.sh --tag $tag --id $id --test_stage $test_stage --stage 6
```
Note that the inference results have differences between the quick test and the strict check under the causal condition due to different paddings.

## Results
([↑up to contents](#contents))

The following results are obtained using strict check `Stage=6`. We use acoustic models trained by the script from [SMS-WSJ](https://github.com/fgnt/sms_wsj). The decode script can be called like
```bash
./asr_decodespwsj2/run_decode.sh $ROOT/ConvTasNet_parcoder2_snr_serial3/exp/train_convtasnet_reverb2reverb_8kmin_823e6963noncausal/examples_strictcheck1bfs
```
We recommend you put `asr_decodespwsj2` into SMS-WSJ directory.

### Noncausal
| Stage 1 | Iteration # on Stage 2 | w/o last MVDR |     | w/ last MVDR |     |
|---------|------------------------|---------------|-----|--------------|-----|
|         |                        | SDR           | WER | SDR          | WER |
| Beam-TasNet |  | 12.652 | 22.11 | 17.387 | 13.38 |
| ✔ | 0 | 11.723 | 25.39 | 16.723 | 13.96 |
| ✔ | 1 | 18.805 | 13.82 | 19.310 | 12.40 |
| ✔ | 2 | 20.917 | 12.88 | 20.063 | 12.06 |
| ✔ | 3 | 21.354 | 12.77 | 20.334 | 12.03 |
| ✔ | 4 | **21.775** | 12.70 | 20.392 | **12.01** |
| ✔ | 5 | 21.534 | 12.78 | 20.268 | 12.08 |
| ✔ | 6 | 21.635 | 12.80 | 20.157 | 12.12 |
| ✔ | 7 | 21.241 | 12.91 | 19.924 | 12.17 |

### Causal

| Stage 1 | Iteration # on Stage 2 | w/o last MVDR |     | w/ last MVDR |     |
|---------|------------------------|---------------|-----|--------------|-----|
|         |                        | SDR           | WER | SDR          | WER |
| Beam-TasNet |  | 9.030 | 33.55 | 11.358 | 21.41 |
| ✔ | 0 | 8.628  | 35.08 | 10.900 | 22.65 |
| ✔ | 1 | 13.058 | 19.74 | 12.237 | 19.95 |
| ✔ | 2 | 13.901 | 18.65 | 12.456 | 19.38 |
| ✔ | 3 | 13.810 | 18.56 | 12.371 | **19.36** |
| ✔ | 4 | **13.988** | 18.56 | 12.327 | 19.43 |
| ✔ | 5 | 13.545 | 18.67 | 12.064 | 19.64 |
| ✔ | 6 | 13.651 | 18.66 | 12.014 | 19.64 |
| ✔ | 7 | 13.166 | 18.89 | 11.717 | 20.05 |

### Oracle test
| Oracle method | Causal | w/o last MVDR |     | w/ last MVDR |     |
|---------------|--------|---------------|-----|--------------|-----|
|               |        | SDR           | WER | SDR          | WER |
| Signal | × | ∞ | 11.67 | 23.481 | 11.89 |
| Mask   | x | 11.004 | 28.09 | 14.458 | 15.75 |
| Mask-avg   | x | 11.004 | 28.09 | 14.711 | 15.01 |
| Signal | ✔ | ∞ | 11.67 | 17.977 | 13.18 |
| Mask   | ✔ | 11.004 | 28.09 | 10.557 | 20.85 |
| Mask-avg | ✔ | 11.004 | 28.09 | 8.637 | 23.24 |

## Acknowledgement
Thanks for `Asteroid` providing the basic training framework,
```BibTex
@inproceedings{Pariente2020Asteroid,
    title={Asteroid: the {PyTorch}-based audio source separation toolkit for researchers},
    author={Manuel Pariente and Samuele Cornell and Joris Cosentino and Sunit Sivasankaran and
            Efthymios Tzinis and Jens Heitkaemper and Michel Olvera and Fabian-Robert Stöter and
            Mathieu Hu and Juan M. Martín-Doñas and David Ditter and Ariel Frank and Antoine Deleforge
            and Emmanuel Vincent},
    year={2020},
    booktitle={Proc. Interspeech},
}
```
Thank for `ESPnet` with the MVDR code,
```BibTex
@inproceedings{li2020espnet,
  title={{ESPnet-SE}: End-to-End Speech Enhancement and Separation Toolkit Designed for {ASR} Integration},
  author={Chenda Li and Jing Shi and Wangyou Zhang and Aswin Shanmugam Subramanian and Xuankai Chang and Naoyuki Kamo and Moto Hira and Tomoki Hayashi and Christoph Boeddeker and Zhuo Chen and Shinji Watanabe},
  booktitle={Proceedings of IEEE Spoken Language Technology Workshop (SLT)},
  pages={785--792},
  year={2021},
  organization={IEEE},
}
```
Thanks for `SMS-WSJ` for the ASR training script,
```BibTex
@Article{SmsWsj19,
  author    = {Drude, Lukas and Heitkaemper, Jens and Boeddeker, Christoph and Haeb-Umbach, Reinhold},
  title     = {{SMS-WSJ}: Database, performance measures, and baseline recipe for multi-channel source separation and recognition},
  journal   = {arXiv preprint arXiv:1910.13934},
  year      = {2019},
}
```

## Citing us
([↑up to contents](#contents))

If you loved this idea and you want to cite us, use this :
```BibTex
@misc{chen2021beamguided,
      title={Beam-Guided TasNet: An Iterative Speech Separation Framework with Multi-Channel Output}, 
      author={Hangting Chen and Pengyuan Zhang},
      year={2021},
      eprint={2102.02998},
      archivePrefix={arXiv},
      primaryClass={eess.AS}
}
```
