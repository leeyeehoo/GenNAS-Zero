# GenNAS-Zero

This repository is the official implementation of Generic Neural Architecture Search via Regression ([NeurIPS'21 spotlight](https://papers.nips.cc/paper/2021/hash/aba53da2f6340a8b89dc96d09d0d0430-Abstract.html) | [Openreview](https://openreview.net/forum?id=mPTfR3Upe0o) | [Arxiv version](https://arxiv.org/abs/2108.01899)). Besides, a faster method is introduced [Arxiv version](https://arxiv.org/abs/2210.09459).

## Todo

- [x] NASBench-101
- [x] Training
- [x] requirement.sh
- [ ] Exploration
- [x] NASBench-201
- [x] NDS
- [x] TransNASBench-Micro
- [x] NASBench-MB
- [ ] Preprocessing data
- [ ] ReadMe
- [x] Standalone Instruction

## Standalone Instruction

Please run `notebooks/standalone.ipynb`

## Start a Simple Zero Cost Proxy Sarch

    
    bash requirement.sh
    conda activate gennaszero
    python do_train.py --config=config_loss_macs_nb101_nas


## References
```
@article{li2021generic,
  title={Generic neural architecture search via regression},
  author={Li, Yuhong and Hao, Cong and Li, Pan and Xiong, Jinjun and Chen, Deming},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  pages={20476--20490},
  year={2021}
}
```

