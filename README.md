# Beam-Guided-TasNet
Beam-Guided TasNet: An Iterative Speech Separation Framework with Multi-Channel Output

Please refer to `https://arxiv.org/abs/2102.02998`

## Contents
- [Beam-Guided-TasNet](#beam-guided-tasnet)
  - [Contents](#contents)
  - [Installation](#installation)
  - [Train](#train)
  - [Test](#test)
  - [Citing us](#citing-us)

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
