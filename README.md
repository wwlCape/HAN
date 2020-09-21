## HAN

> PyTorch code for our ECCV 2020 paper "Single Image Super-Resolution via a Holistic Attention Network"
>
> This repository is for HAN introduced in the following paper
>
> Yulun Zhang, Kunpeng Li, Kai Li, Lichen Wang, Bineng Zhong, and Yun Fu, "Single Image Super-Resolution via a Holistic Attention Network", ECCV 2020, [arxiv](https://arxiv.org/abs/2008.08767)
>
> The code is built on RCAN (PyTorch) and tested on Ubuntu 16.04/18.04 environment (Python3.6, PyTorch_0.4.0, CUDA8.0, cuDNN5.1) with Titan X/1080Ti/Xp GPUs.
>
> ### Contents
>
> ________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________
>
> > 1. [Introduction](https://github.com/wwlCape/HAN#introduction)
> > 2. [Train](https://github.com/wwlCape/HAN#Begin to train)
> > 3. [Test](https://github.com/wwlCape/HAN#Begin to test)
> > 4. [Acknowledgements](https://github.com/wwlCape/HAN#Acknowledgements)
>
> ### Introduction
>
> Informative features play a crucial role in the single image super-resolution task. Channel attention has been demonstrated to be effective for preserving information-rich features in each layer. However, channel attention treats each convolution layer as a separate process that misses the correlation among different layers. To address this problem, we propose a new holistic attention network (HAN), which consists of a layer attention module (LAM) and a channel-spatial attention module (CSAM), to model the holistic interdependencies among layers, channels, and positions. Specifically, the proposed LAM adaptively emphasizes hierarchical features by considering correlations among layers. Meanwhile, CSAM learns the confidence at all the positions of each channel to selectively capture more informative features. Extensive experiments demonstrate that the proposed HAN performs favorably against the state-of-the-art single image super- resolution approaches.
>
>
> Train
> Prepare training data
> Download DIV2K training data (800 training + 100 validtion images) from DIV2K dataset.
>
> ### Begin to train
>
> (optional) Download models for our paper and place them in '/HAN/experiment/HAN'. All the models (BIX2/3/4/8, BDX3) can be downloaded from [GoogleDrive](https://drive.google.com/drive/folders/17cLcPCDLuBV5_5-ngd0vXIDp6rebIMG1). You can use scripts in file 'demo.sh' to train models for our paper.
>
> ```python
> BI, scale 2, 3, 4, 8
> #HAN BI model (x2)
> 
> python main.py --template HAN --save HANx2 --scale 2 --reset --save_results --patch_size 96 --pre_train ../experiment/model/RCAN_BIX2.pt
> 
> #HAN BI model (x3)
> 
> python main.py --template HAN --save HANx3 --scale 3 --reset --save_results --patch_size 144 --pre_train ../experiment/model/RCAN_BIX2.pt
> 
> #HAN BI model (x4)
> 
> python main.py --template HAN --save HANx4 --scale 4 --reset --save_results --patch_size 192 --pre_train ../experiment/model/RCAN_BIX2.pt
> 
> #HAN BI model (x8)
> 
> python main.py --template HAN --save HANx8 --scale 8 --reset --save_results --patch_size 384 --pre_train ../experiment/model/RCAN_BIX2.pt
> 
> 
> ```
>
> ### Begin to Test
>
> ```python
> Quick start
> 
> Download models for our paper and place them in '/experiment/HAN'.
> 
> Cd to '/HAN/src', run the following scripts.
> #test
> python main.py --template HAN --data_test Set5+Set14+B100+Urban100+Manga109 --data_range 801-900 --scale 2 --pre_train ../experiment/HAN/HAN_BIX2.pt --test_only --save HANx2_test --save_results
> ```
>
> All the models (BIX2/3/4/8, BDX3) can be downloaded from [GoogleDrive](https://drive.google.com/drive/folders/17cLcPCDLuBV5_5-ngd0vXIDp6rebIMG1).
>
> The whole test pipeline 
>
> 1.Prepare test data.
>
> Place the original test sets in '/dataset/x4/test'.
>
> Run 'Prepare_TestData_HR_LR.m' in Matlab to generate HR/LR images with different degradation models.
>
> 2.Conduct image SR.
>
> See Quick start
>
> 3.Evaluate the results.
>
> Run 'Evaluate_PSNR_SSIM.m' to obtain PSNR/SSIM values for paper.
>
> ### Acknowledgements
>
> This code is built on [RCAN](https://github.com/yulunzhang/RCAN). We thank the authors for sharing their codes of RCAN  [PyTorch version](https://github.com/yulunzhang/RCAN).

