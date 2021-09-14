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

You can give a quick test by running :
```bash
./run.sh --tag $tag --id $id --test_stage $test_stage --stage 5
```
You can give a strict check by running :
```bash
./run.sh --tag $tag --id $id --test_stage $test_stage --stage 6
```
Note that the inference results have differences between the quick test and the strict check under the causal condition due to different paddings. Besides, the strict check is slow due to large redundant computations in each frame.

## Results
([↑up to contents](#contents))

The following results are obtained using `Stage=6`. We use acoustic models trained by the script from [SMS-WSJ](https://github.com/fgnt/sms_wsj).

### Noncausal
| Stage 1 | Iteration # on Stage 2 | w/o last MVDR |     | w/ last MVDR |     |
|---------|------------------------|---------------|-----|--------------|-----|
|         |                        | SDR           | WER | SDR          | WER |
| ✔ | 0 | 11.823 | 25.08 | 16.768 | 13.83 |
| ✔ | 1 | 18.738 | 13.50 | 19.181 | 12.39 |
| ✔ | 2 | 19.727 | 13.08 | 19.735 | **12.23** |
| ✔ | 3 | **19.930** | 12.98 | 19.645 | 12.37 |
| ✔ | 4 | 19.442 | 13.19 | 19.203 | 12.61 |

### Causal
| Stage 1 | Iteration # on Stage 2 | w/o last MVDR |     | w/ last MVDR |     |
|---------|------------------------|---------------|-----|--------------|-----|
|         |                        | SDR           | WER | SDR          | WER |
| ✔ | 0 | 8.356 | 37.20 | 10.558 | 23.79 |
| ✔ | 1 | 12.401 | 21.83 | 11.612 | 21.67 |
| ✔ | 2 | **13.016** | **20.61** | 11.710 | 21.32 |
| ✔ | 3 | 12.961 | 20.80 | 11.607 | 21.67 |
| ✔ | 4 | 12.932 | 21.02 | 11.456 | 21.85 |

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
