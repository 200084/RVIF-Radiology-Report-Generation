# RVIF-Radiology-Report-Generation
### Implementation of Reinforced Visual Interaction Fusion Radiology Report Generation (RVIF). [[Multimedia Systems 2024](https://link.springer.com/article/10.1007/s00530-024-01504-8)]
<div align="center">
  <img src="https://github.com/200084/RVIF-Radiology-Report-Generation/blob/main/RVIF.jpg">
</div>

# Requirements 
## Enviroment & Dataset Preparation 
Enviroment and dataset are both reference to [R2GenCMN](https://github.com/zhjohnchan/R2GenCMN)
The dataset contains IU X-Ray and MIMIC-CXR.

# Training
## Training for the RVIF-NO-RL
Run bash `train_iu_xray.sh` in `/RVIF-NO-RL` to test on the IU X-Ray data.
Run bash `train_mimic_cxr.sh` in `/RVIF-NO-RL` to test on the MIMIC-CXR data.
## Training for the RVIF-RL
Run bash `run_rl_iu.sh` in `/RVIF-RL` to test on the IU X-Ray data.
Run bash `run_rl_mimic` in `/RVIF-RL` to test on the MIMIC-CXR data.

# Evaluation or Testing 
## NLG Evaluation or Testing
Run bash `run_rl_iu.sh` in `/RVIF-RL` to test on the IU X-Ray data. Both training and test metrics are included in the resulting `CSV`.
Run bash `run_rl_mimic.sh` in `/RVIF-RL` to test on the MIMIC-CXR data. Both training and test metrics are included in the resulting `CSV`.
## CE Evaluation or Testing
Follow [CheXpert](https://github.com/MIT-LCP/mimic-cxr/tree/master/txt/chexpert) or [CheXbert](https://github.com/stanfordmlgroup/CheXbert) to extract the labels and then run python `compute_ce.py`. Note that there are several steps that might accumulate the errors for the computation, e.g., the labelling error and the label conversion. 

# Reference
If you find this repo useful, please consider citing (no obligation at all):
```
@article{wang2024reinforced,
  title={Reinforced visual interaction fusion radiology report generation},
  author={Wang, Liya and Chen, Haipeng and Liu, Yu and Lyu, Yingda and Qiu, Feng},
  journal={Multimedia Systems},
  volume={30},
  number={5},
  pages={299},
  year={2024},
  publisher={Springer}
}
```
