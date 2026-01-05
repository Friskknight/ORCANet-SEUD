# All-in-One Video Restoration under Smoothly Evolving Unknown Weather Degradations
**Authors:** Wenrui Li, Hongtao Chen, Yao Xiao, Wangmeng Zuo (Senior Member, IEEE), Jiantao Zhou (Senior Member, IEEE), Yonghong Tian (Fellow, IEEE), Xiaopeng Fan (Senior Member, IEEE)

[![arXiv](https://img.shields.io/badge/arXiv-2601.00533-b31b1b.svg)](https://arxiv.org/abs/2601.00533)

---

- [2026/01] ORCANet training and inference code is released.
- [2026/01] The paper is available on [arXiv](https://arxiv.org/abs/2601.00533).
- Synthetic SEUD data generation code and dataset are under preparation.

---

## Overview
This repository provides the official implementation of **ORCANet** for all-in-one video restoration under smoothly evolving and unknown weather degradations (SEUD). ORCANet integrates coarse intensity–aware dehazing and a temporal prompt design for stable restoration under evolving degradation types and intensities.
<p align="center">
  <img src="assets/mainflowcrop.jpg" width="100%">
</p>

**Figure:** Overview of the ORCANet.

---

## Environment Setup

### 1) Install dependencies
```bash
git clone https://github.com/Friskknight/ORCANet-SEUD.git
cd ORCANet-SEUD
pip install -r requirement.txt
python setup.py develop
```

---

## Preparation

### 1) Download experiment folder (pretrained weights)

Download the `experiment` folder from Google Drive and place it in the repository root directory:

- Google Drive: https://drive.google.com/drive/folders/1b6IpTLh-SJOuDo8oYSREKRpLY5zhxpiq?usp=sharing

After downloading, your directory should look like:

```
ORCANet-SEUD/
  experiment/pretrained_models
  basicsr/
  options/
  scripts/
  ...
```

### 2) Prepare your own dataset (optional)

The synthesis code and the SEUD dataset in paper are under preparation and will be released in a later update. If you use your own training data, generate the meta information file:

```
python scripts/data_preparation/generate_meta_info.py --dataset_path "The root of training sets"
```

The generated file is saved to:

```
basicsr/data/meta_info/<your_dataset>_meta_info.txt
```

---

## SEUD Synthesis Code and Dataset

The synthesis code and the SEUD dataset are under preparation and will be released in a later update.

<p align="center">
  <img src="assets/synthsis_crop.jpg" width="100%">
</p>


**Figure:** Overview of the SEUD weather synthesis pipeline.

---

## Training

Before training, please check the YAML file:

- `dataroot_gt`
- `dataroot_lq`
- `meta_info_file`

Example (single GPU):

```
python basicsr/train.py -opt options/train/train_ORCANet_MOT17t4.yml
```

Multi-GPU training:

```
CUDA_VISIBLE_DEVICES=0,1 LD_LIBRARY_PATH="" torchrun --nproc_per_node=2 --master_port=4321 \
  basicsr/train.py -opt options/train/train_ORCANet_MOT17t4.yml --launcher pytorch
```

------

## Inference (Testing)

Before testing, please check the YAML file:

- `dataroot_gt`
- `dataroot_lq`
- `meta_info_file`

You can enable/disable saving outputs via `save_img` in the test YAML.

Run inference:

```
python basicsr/test.py -opt options/test/test_ORCANet_MOT17t4.yml
```

------

## Acknowledgements

This project builds upon and is inspired by the following open-source projects and resources:

- BasicSR: https://github.com/XPixelGroup/BasicSR
- AverNet: https://github.com/XLearning-SCU/2024-NeurIPS-AverNet/tree/main
- Depth Anything: https://github.com/LiheYoung/Depth-Anything
- IntoTheFog: https://github.com/nadezola/IntoTheFog_MOT17/tree/master

We thank the authors for their excellent work.

------

## Contact

If you have any questions, please contact ht166chen@163.com