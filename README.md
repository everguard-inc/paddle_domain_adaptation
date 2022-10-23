# Unsupervised Domain Adaptation on Semantic Segmentation

Domain adaptation is the ability to apply an algorithm trained in one or more "source domains" to a different (but related) "target domain". With domain adaptation algorithms, performance drop caused by [domain shift](https://en.wikipedia.org/wiki/Domain_adaptation#:~:text=A%20domain%20shift%2C%20or%20distributional,practical%20applications%20of%20artificial%20intelligence.) can be alleviated. Specifically, none of the manually labeled images will be used in unsupervised domain adaptation(UDA). The following picture shows the result of applying our  unsupervised domain adaptation algorithms on semantic segmentation task. By comparing the segmentation results between "without DA" and "with DA", we can observe a remarkable performance gain.

<p align="center">
<img src="https://raw.githubusercontent.com/PaddlePaddle/PaddleSeg/a73f874019afb5a36aa5cb60131b834282a91c25/contrib/DomainAdaptation/docs/domain_adaptation.png" width="100%" height="100%">
</p>

In this project, we reproduce PixMatch [[Paper](https://arxiv.org/abs/2105.08128)|[Code](https://github.com/lukemelas/pixmatch)] with PaddlePaddle and reaches mIOU = 47.8% on Cityscapes Dataset.

On top of that, we also tried several adjustments including:

1. Add edge constrain branch to improve edge accuracy (negative results, still needs to adjust)
2. Use edge as prior information to fix segmentation result (negative results, still needs to adjust)
3. Align features' structure across domain (positive result, reached mIOU=48.0%)

### 1. Install environment

```
git clone https://github.com/PaddlePaddle/PaddleSeg.git
cd contrib/DomainAdaptation/
pip3 install -r requirements.txt
python3 -m pip install paddlepaddle-gpu==2.2.0 -i https://mirror.baidu.com/pypi/simple
```

### 3. Train and test

1. Train on one GPU
   ```
   python3 train.py --config /home/eg/rodion/paddle_domain_adaptation/configs/deeplabv2/custom.yml
   ```
2. Train on multiple GPU
   ```
   python3 -m paddle.distributed.launch train.py --config /home/eg/rodion/paddle_domain_adaptation/configs/deeplabv2/custom.yml --num_workers 4
   ```

3. Validate :
   ```
   python3 -m paddle.distributed.launch val.py --config --config /home/eg/rodion/paddle_domain_adaptation/configs/deeplabv2/custom.yml --model_path models/model.pdparams --num_workers 4
   ```
