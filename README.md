# End-to-End Face Parsing via Interlinked Convolutional Neural Networks

offical STN-iCNN codes

### TOC

[TOC]

## Training

There are three components in our framework in STN-iCNN framework:

- modelA: the Stage1 iCNN (global seg)

- modelB: the STN-based affine cropper

- modelC: the stage2 iCNN (local seg)

### Train modelA

```shell
python3 train_stage1.py --cuda 0 --datamore 1 --lr 0.0009
```
The best training file will be found in checkpointsA/{this training uuid}/best.pth.tar

Our trainning results are:

```shell
epoch 25        error 0.0254    best_error 0.0235
training ID: 48fb8cd4 
The f1 overall for modelA seg result is:
0.8572761364675753
```

[Download checkpoint 48fb8cd4 at here.](https://github.com/aod321/MyModelZoo/raw/main/stn_icnn_pretrains/48fb8cd4.pth.tar)

### Train modelA+B (facial parts cropper)

change the path_modelA to yours at the line 282 in 2_tuning_cropper_model1.py:

```
path_modelA = os.path.join("/home/yinzi/data4/STN-iCNN/checkpoints_A/48fb8cd4", 'best.pth.tar')
```

the run training:

```shell
python3 2_tuning_cropper_model1.py --pretrainA 1 --pretrainB 0 --select_net 1 --lr 0 --lr_s 0.0008 --cuda 7 --batch_size 32 --epochs 3000
```

The best training file will be found in checkpointsAB_res/{this training uuid}/best.pth.tar

Our trainning results are:

```shell
checkpoints_AB_res/e0de5954
best_error 1.53e-05
```

![image.png](https://i.loli.net/2021/08/31/FUoCNIqkwJDyxG3.png)

[Download checkpoint e0de5954 at here.](https://github.com/aod321/MyModelZoo/raw/main/stn_icnn_pretrains/e0de5954.pth.tar)

### Train modelC (the model for facial parts segmentation)

```shell
python3 train_stage2.py --batch_size 64 --cuda 6 --lr0 0.0008 --lr1 0.0008 --lr2 0.0008 --lr3 0.0008 --epochs 3000
```

Our trainning results are:

```
c1f2ab1a
best_error_all 0.245
```

[Download checkpoint c1f2ab1a at here.](https://github.com/aod321/MyModelZoo/raw/main/stn_icnn_pretrains/c1f2ab1a.pth.tar)

### End2End finetuning

Change the path for checkpoint e0de5954 and checkpoint c1f2ab1a at line 291 in the 3_end2end_tunning_all.py, then run

```shell
python3 3_end2end_tunning_all.py --batch_size 32 --cuda 8 --select_net 1 --pretrainA 1 --pretrainB 1 --pretrainC 0 --lr 0 --lr2 0.0025 --lr_s 0 --epoch 3000 --f1_eval 1
```

Our trainning results are:

![image.png](https://i.loli.net/2021/08/31/s9jmgNERfwUrObz.png)

## Test

Loading parmartes from checkpoint e0de5954(for model A+B) and checkpoint c1f2ab1a(for modelC), without any finetune, we tested our results, the results can be found in this jupyter notebook.

https://github.com/aod321/STN-iCNN/blob/start_again/before_end2end_fintune_test.ipynb

And the results after end2end finetune can be found in this jupyter notebook.

https://github.com/aod321/STN-iCNN/blob/start_again/after_end2end_fintune_test.ipynb

Note:

We found that the f1 score after finetune reported in our paper (0.91) cannot be reproduced stably: under the same hyperparameter settings, it can occasionally reach that result, but after training 10 times and taking the average, we only get 0.90. The stable result is 1% decay compared with the result reported in our paper. 



## Pretrained End2End Model

Before end2end

[Download at here.](https://github.com/aod321/MyModelZoo/blob/main/stn_icnn_pretrains/before_end2end.pth.tar)

After end2end

[Download at here.](https://github.com/aod321/MyModelZoo/blob/main/stn_icnn_pretrains/after_end2_end_10mean.pth.tar)


