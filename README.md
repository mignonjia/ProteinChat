# Multi-Modal Large Language Model Enables Protein Function Prediction

This repository holds the code and data of Multi-Modal Large Language Model Enables Protein Function Prediction.

## Examples

![Eg1](fig/example.pdf) 


## Introduction
- ProteinChat is a versatile, multi-modal large language model designed to predict protein functions from amino acid sequences.
- ProteinChat works in a similar way as ChatGPT. Users upload a protein sequence and ask various questions about this protein. ProteinChat will answer these questions in a multi-turn, interactive manner. 
- The ProteinChat system consists of a protein sequence encoder (based on [xTrimoGLM](https://arxiv.org/abs/2401.06199)), a large language model (LLM), and an adaptor. The protein encoder takes a protein sequence as input and learns a representation for this protein. The adaptor transforms the protein representation produced by the protein encoder  into another  representation that is acceptable to the  LLM. The LLM takes the representation transformed by the adaptor and users' questions about this protein as inputs and generates answers. All these components are trained end-to-end.
- To train ProteinChat, we designed (protein, prompt, answer) triples from the functions and keywords from Swiss-Prot dataset, resulting in 462,019 proteins and 1.5 million Q&A pairs.


![overview](fig/workflow.pdf)

## Prepare the Dataset

The dataset contains 462,019 proteins (represented using 3D structures) with 1.5 million instructions. It is curated from the [Swiss-Prot Dataset](https://www.uniprot.org/uniprotkb?query=*&facets=reviewed%3Atrue). 
The dataset `data.tar.gz` (148 MB) can be downloaded [here](https://drive.google.com/file/d/1n5Ant3S5QE0Yx-DznRa3lannFanc1WB7/view?usp=sharing). Copy it under this folder and run 
```bash
tar -xvf data.tar.gz
```

## Getting Started
### Installation

**1. Prepare the code and the environment**

Git clone our repository, creating a python environment and ativate it via the following command

```bash
git clone https://github.com/UCSD-AI4H/proteinchat
cd proteinchat
conda env create -f environment.yml
conda activate proteinchat
```

Verify the installation of `torch` and `torchvision` is successful by running `python -c "import torchvision; print(torchvision.__version__)"`. If it outputs the version number without any warnings or errors, then you are good to go. __If it outputs any warnings or errors__, try to uninstall `torch` by `conda uninstall pytorch torchvision torchaudio cudatoolkit` and then reinstall them following [here](https://pytorch.org/get-started/previous-versions/#v1121). You need to find the correct command according to the CUDA version your GPU driver supports (check `nvidia-smi`). 

**2. Prepare the pretrained Vicuna weights**

The current version of ProteinChat is built on Vicuna-13B-v1.5.
Please download Vicuna weights from [https://huggingface.co/lmsys/vicuna-13b-v1.5](https://huggingface.co/lmsys/vicuna-13b-v1.5).
Then, set the path to the vicuna weight in the config files 
[configs/proteinchat_stage1.yaml](configs/proteinchat_stage1.yaml#L15) and [configs/proteinchat_stage1.yaml](configs/proteinchat_stage2.yaml#L15).


### Training
**You need at least 55 GB GPU memory for the training.** 

The stage-1 training configuration file is [configs/proteinchat_stage1.yaml](configs/proteinchat_stage1.yaml). In addition, you may want to change the number of epochs and other hyper-parameters there, such as `max_epoch`, `init_lr`, `min_lr`,`warmup_steps`, `batch_size_train`. Please adjust `iters_per_epoch` so that `iters_per_epoch` * `batch_size_train` = your training set size. Due to the GPU consumption, we set `batch_size_train=1`. 

Also, set your desired output directory [here](configs/proteinchat_stage1.yaml#52).

Start stage-1 training by running 
```bash
bash finetune.sh --cfg-path configs/proteinchat_stage1.yaml
``` 

The stage-2 training configuration file is [configs/proteinchat_stage2.yaml](configs/proteinchat_stage2.yaml). Replace the `stage1_ckpt` with the checkpoint you obtained in stage 1. Similar with the previous step, you also need to replace the output directory in this file.

Start stage-2 training by running 
```bash
bash finetune.sh --cfg-path configs/proteinchat_stage2.yaml
``` 

### Evaluation

**It takes around 24 GB GPU memory for the inference.**

Modify the checkpoint paths in [configs/proteinchat_eval.yaml](configs/proteinchat_eval.yaml) to the location of your checkpoint.

Start evaluation on 200 samples in the test set by running 
```bash
bash demo.sh
``` 


## Acknowledgement

+ [xTrimoPGLM](https://arxiv.org/abs/2401.06199)
+ [ESM](https://github.com/facebookresearch/esm)
+ [MiniGPT-4](https://minigpt-4.github.io/) 
+ [Lavis](https://github.com/salesforce/LAVIS)
+ [Vicuna](https://github.com/lm-sys/FastChat)


## License
This repository is under [BSD 3-Clause License](LICENSE.md).


## Citation

If you're using ProteinChat in your research or applications, please cite using this BibTeX:
```bibtex
@article{huo2024multi,
  title={Multi-Modal Large Language Model Enables Protein Function Prediction},
  author={Guo, Han and Huo, Mingjia and Xie, Pengtao},
  year={2024}
}
```

