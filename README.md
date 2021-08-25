# Towards Improved and Interpretable Deep Metric Learning via Attentive Grouping
This repository is an official PyTorch implementation of DGML in "Towards Improved and Interpretable Deep Metric Learning via Attentive Grouping". For more insights, please refer to our paper.

[Xinyi Xu](https://xinyixuxd.github.io/), [Zhengyang Wang](https://zhengyang-wang.github.io/), [Cheng Deng](https://see.xidian.edu.cn/faculty/chdeng/), [Hao Yuan](https://sites.google.com/site/hyuanustc), and [Shuiwang Ji](http://people.tamu.edu/~sji/). [Towards Improved and Interpretable Deep Metric Learning via Attentive Grouping](https://arxiv.org/pdf/2011.08877.pdf)

## Reference
```
@article{xu2020towards,
  title={Towards Improved and Interpretable Deep Metric Learning via Attentive Grouping},
  author={Xu, Xinyi and Wang, Zhengyang and Deng, Cheng and Yuan, Hao and Ji, Shuiwang},
  journal={arXiv preprint arXiv:2011.08877},
  year={2020}
}
```

## Requirements
* Pytorch
* yacs
* numpy
* matplotlib
* tqdm
* termcolor

## Run
* To reproduce our results in Table 2, run 
```linux
CUDA_VISIBLE_DEVICES=0 python main.py dataset 'cub200' or

CUDA_VISIBLE_DEVICES=0 python main.py dataset 'cars196' or

CUDA_VISIBLE_DEVICES=0 python main.py dataset 'online_products'
```

* We also provide the results of in_shop dataset which is not given in our paper. We achive 88.93% of recall@1 evaluation metric. To get the results, run
```linux
CUDA_VISIBLE_DEVICES=0 python main.py dataset 'in_shop'
```
## Note.

This repository contains some codes from: https://github.com/Confusezius/Deep-Metric-Learning-Baselines

