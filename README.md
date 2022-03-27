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

Requirement installation:
```bash
pip install -r requirements.txt
```

This recipe requires a modified version of `asteroid-filterbanks`. 
The script `run.sh` will automatically download these requirements in the current directory.

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
./run.sh --id 0,1,2,3
```

## Test
([↑up to contents](#contents))

You can give a strict check by running :
```bash
./run.sh --tag $tag --id $id --test_stage $test_stage --stage 6
```

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
| ✔ | 0 | 10.519 | 29.76 | 15.874 | 14.79 |
| ✔ | 1 | 18.210 | 14.03 | 19.132 | 12.33 |
| ✔ | 2 | 20.666 | 12.88 | 19.959 | 12.12 |
| ✔ | 3 | 21.334 | 12.76 | 20.236 | 12.07 |
| ✔ | 4 | **21.529** | 12.78 | 20.282 | **12.09** |
| ✔ | 5 | 21.527 | 12.73 | 20.213 | **12.09** |
| ✔ | 6 | 21.419 | 12.81 | 20.078 | 12.15 |
| ✔ | 7 | 21.253 | 12.83 | 19.904 | 12.21 |

### Causal

| Stage 1 | Iteration # on Stage 2 | w/o last MVDR |     | w/ last MVDR |     |
|---------|------------------------|---------------|-----|--------------|-----|
|         |                        | SDR           | WER | SDR          | WER |
| Beam-TasNet |  | 9.030 | 33.55 | 11.358 | 21.41 |
| ✔ | 0 | 8.628  | 35.08 | 10.900 | 22.65 |
| ✔ | 1 | 13.058 | 19.74 | 12.237 | 19.95 |
| ✔ | 2 | 13.901 | 18.65 | 12.456 | 19.38 |
| ✔ | 3 | 13.810 | **18.56** | 12.371 | 19.36 |
| ✔ | 4 | **13.988** | **18.56** | 12.327 | 19.43 |
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
