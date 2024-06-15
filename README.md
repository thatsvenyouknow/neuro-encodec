# Neuro-Encodec
This repository is forked from the great EnCodec repository associated with the [High Fidelity Neural Audio Compression](https://arxiv.org/pdf/2210.13438.pdf) paper that aims to compress audio.
We have added some functionality to train/finetune the EnCodec model for its use in the [Neuralink Compression Challenge](https://content.neuralink.com/compression-challenge/README.html).


## Installation

Python 3.8 is required and a reasonably recent version of PyTorch (1.11.0 ideally).

To install (Neuro-)EnCodec, you can run from this repository:

1. Clone repo locally
2. 
```bash
pip install .
```


### Distributed Training

The train script does not support distributed training, yet. Soon to come...

## Acknowledgment
We want to thank the following open-source repositories on which we based this project:
- [EnCodec](https://github.com/facebookresearch/encodec)
- [encodec-pytorch](https://github.com/ZhikangNiu/encodec-pytorch)

## License

The code in this repository is released under the MIT license as found in the
[LICENSE](LICENSE) file.
