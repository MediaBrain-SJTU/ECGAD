# Multi-scale Cross-restoration Framework for Electrocardiogram Anomaly Detection

This repository contains the official PyTorch implementation of

[**Multi-scale Cross-restoration Framework for Electrocardiogram Anomaly Detection</a>**](http://arxiv.org/abs/2308.01639)

Aofan Jiang, [Chaoqin Huang](https://scholar.google.com/citations?user=BAZSE7wAAAAJ&hl=en), Qing Cao, Shuang Wu, Zi Zeng, Kang Chen, [Ya Zhang](https://scholar.google.com/citations?user=pbjw9sMAAAAJ&hl=en), and Yanfeng Wang

Presented at [MICCAI 2023](https://conferences.miccai.org/2023/en/) (Early Accept)

```
@inproceedings{jiang2022ecgad,
  title={Multi-scale Cross-restoration Framework for Electrocardiogram Anomaly Detection}
  author={Jiang, Aofan and Huang, Chaoqin and Cao, Qing and Wu, Shuang and Zeng, Zi and Chen, Kang and Zhang, Ya and Wang, Yanfeng},
  booktitle={International Conference on Medical Image Computing and Computer Assisted Intervention (MICCAI)},
  year={2023}
}
```


<center><img src="figures/pipeline.png "width="90%"></center>

**Abstract**:  Electrocardiogram (ECG) is a widely used diagnostic tool for detecting heart conditions. Rare cardiac diseases may be underdiagnosed using traditional ECG analysis, considering that no training dataset can exhaust all possible cardiac disorders. This paper proposes using anomaly detection to identify any unhealthy status, with normal ECGs solely for training. However, detecting anomalies in ECG can be challenging due to significant inter-individual differences and anomalies present in both global rhythm and local morphology. To address this challenge, this paper introduces a novel multi-scale cross-restoration framework for ECG anomaly detection and localization that considers both local and global ECG characteristics. The proposed framework employs a two-branch autoencoder to facilitate multi-scale feature learning through a masking and restoration process, with one branch focusing on global features from the entire ECG and the other on local features from heartbeatlevel details, mimicking the diagnostic process of cardiologists. Anomalies are identified by their high restoration errors. To evaluate the performance on a large number of individuals, this paper introduces a new challenging benchmark with signal point-level ground truths annotated by experienced cardiologists.

**Keywords**: Anomaly Detection, Electrocardiogram


## Get Started

### Environment
- python >= 3.6.9
- pytorch >= 1.10.0
- torchvision >= 0.11.1
- numpy >= 1.19.2
- scipy >= 1.5.2
- matplotlib >= 3.3.2
- wfdb >= 4.1.0
- heartpy >= 1.2.7
- tqdm


### Files Preparation

1. Download the PTB-XL dataset [here](https://physionet.org/content/ptb-xl/1.0.3/) and preprocess it by `preprocess.py`. Or one can choose to directly download our processed PTB-XL data set for anomaly detection on [Google Drive](https://drive.google.com/drive/folders/1FFPM97nqmTSCx5if18mSDqWvtTCRQfJp?usp=sharing) or [Baidu Disk](https://pan.baidu.com/s/1HdZaMjFrFLvr1Ac-AJrrZw?pwd=hrsh) (hrsh)  and put it under `data` folder.
2. Download our proposed benchmark with signal point-level localization annotations on [Google Drive](https://drive.google.com/drive/folders/1STvUtzuOUovFT4YiQR0H-KTaZr0YnPZ-?usp=sharing) or [Baidu Disk](https://pan.baidu.com/s/1Ew5pSbqBRlUvXSghaxHywg?pwd=439w) (439w) and put it under `data` folder.
3. Download the pre-train models on [Google Drive](https://drive.google.com/file/d/16fGl6G_DOyguNVkBoUwmXK0YiYc21XaA/view?usp=sharing) or [Baidu Disk](https://pan.baidu.com/s/1ccyvR06XYUvXnS6jaEDLDQ?pwd=m60k) (m60k) and put it under `ckpt` folder.

  After the preparation work, the whole project should have the following structure:

  ```
  ./ECGAD
  ├── README.md
  ├── train.py                                  # training code
  ├── test.py                                   # testing code
  ├── ckpt                 
  │   └── mymodel.pt
  ├── data
  │   ├── benchmark_data.npy                    # proposed benchmark data
  │   ├── benchmark_label.npy                   # proposed benchmark label
  │   ├── train.npy                             # processed PTB-XL training data
  │   ├── test.npy                              # processed PTB-XL testing data
  │   └── label.npy                             # processed PTB-XL testing label
  ├── model.py
  ├── dataloader.py               
  └── utils.py
  ```

### Quick Start

```python
python test.py --data_path ./data/ --load_model 1 --load_path ./ckpt/mymodel.pt
```

## Training

```python
python train.py --data_path ./data/ --save_model 1 --save_path ./ckpt/mymodel_new.pt
```

Then you can run the evaluation using:
```python
python test.py --data_path ./data/ --load_model 1 --load_path ./ckpt/mymodel_new.pt
```

## Results

<div style="text-align: center;">
<table>
<tr><td></td> <td colspan="2">Detection (PTB-XL)</td> <td colspan="2">Localization (Benchmark)</td></tr>
<tr><td>AUC (%)</td> <td>Paper</td> <td>Inplementation</td> <td>Paper</td> <td>Inplementation</td></tr>
<tr height='21' style='mso-height-source:userset;height:16pt' id='r0'>
<td height='21' class='x21' width='90' style='height:16pt;width:67.5pt;'>Ours</td>
<td class='x23' width='90' style='width:67.5pt;'>86.0</td>
<td class='x22' width='90' style='width:67.5pt;'><b>86.0</b></td>
<td class='x23' width='90' style='width:67.5pt;'>74.7</td>
<td class='x22' width='90' style='width:67.5pt;'><b>75.4</b></td>
 </tr>
</table>
</div>

## Visualization
<center><img src="figures/vis.png "width="90%"></center>

## Contact

If you have any problem with this code, please feel free to contact **stillunnamed@sjtu.edu.cn**.