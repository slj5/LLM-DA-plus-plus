# A Unified Framework of Data Augmentation Using Large Language Models for Text-based Cross-modal Retrieval

## Set up environment

* Clone the repository: `git clone https://github.com/XinhaoMei/audio-text_retrieval.git`
* Create conda environment with dependencies: `conda env create -f environment.yaml -n name`
* All of our experiments are running on RTX 3090 with CUDA11. This environment just works for RTX 30x GPUs.

## Set up dataset 

* Clotho can be downloaded at https://zenodo.org/record/4783391#.YkRHxTx5_kk.
* Unzip downloaded files, and put the wav files under 'data/Clotho/waveforms`, the folder structured like
```
  data
  ├── Clotho
  │   ├── csv_files  
  │   ├── waveforms
  │      ├── train
  │      ├── val
  │      ├── test
  
  ```

## Pre-trained encoders
* Pre-trained audio encoders ResNet38 can be downloaded at: https://github.com/qiuqiangkong/audioset_tagging_cnn
* Name the pre-trained models to `ResNet38.pth`, and put them under the folder `pretrained_models/audio_encoder` (first create these two folders)

### Run experiments
* Set the parameters you want in `settings/settings.yaml` 
* Run experiments: `python train.py`
