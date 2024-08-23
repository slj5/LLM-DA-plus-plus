# Text to Motion Retrieval


### 1. Data preparation

```
**KIT** - Download from [HumanML3D](https://github.com/EricGuo5513/HumanML3D.git) (no processing needed this time) and the place result in `./dataset/KIT-ML`

**Compute Text Similarities** - This is needed to pre-compute the relevances for NDCG metric to use during validation and testing.
<!-- You have to download the precomputed text similarities from ... and place them under `outputs/computed_relevances`. -->
You can compute them by running the following command:
```
python text_similarity_utils/compute_relevance --set [val|test] --method spacy --dataset [kit]
```

### 2. Train

Run the command```
bash reproduce_train.sh
```
Modify che code appropriately for including/excluding models or loss functions.
This code will create a folder `./runs` where checkpoints and training metrics (tensorboard logs) are stored for each model.

### 3. Test

Run the command
```
bash reproduce_eval.sh
```
Modify che code appropriately following the models and loss functions included in `reproduce_train.sh`. Non-existing configurations will be skipped.
