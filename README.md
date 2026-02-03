# Lᴍᴛᴇ: Putting the "Reasoning" into WAN Traffic Engineering with Language Models <br><sub>Official PyTorch Implementation</sub>

<p align="center">
    <a href= "https://github.com/Y-debug-sys/LMTE/stargazers/">
        <img src="https://img.shields.io/github/stars/Y-debug-sys/LMTE.svg" /></a>
    <a href= "https://github.com/Y-debug-sys/LMTE/network/">
        <img src="https://img.shields.io/github/forks/Y-debug-sys/LMTE.svg" /></a>
    <a href= "http://arxiv.org/abs/2602.00941">
        <img src="https://img.shields.io/badge/Paper-arXiv-red" /></a>
    <a href= "https://huggingface.co/meta-llama/">
        <img src="https://img.shields.io/badge/Hugging%20Face-LLaMA3-teal" /></a>
    <a href= "https://www.python.org/downloads/release/python-31019/">
        <img src="https://img.shields.io/badge/Python-3.10-blue.svg" /></a>
    <a href= "https://github.com/Y-debug-sys/LMTE/blob/master/LICENSE">
        <img src="https://img.shields.io/badge/License-Apache2.0-red.svg" /></a>
    <a href= "https://pytorch.org/">
        <img src="https://img.shields.io/badge/Pytorch-2.7-orange.svg" /></a>
</p>

---

<p align="center">
    <img src='figs/theme.jpg' width='100%' align=center />
</p>

> **Authors:** [Xinyu Yuan](https://y-debug-sys.github.io/), [Yan Qiao](https://faculty.hfut.edu.cn/qiaoyan/en/index.htm), [Zonghui Wang](https://person.zju.edu.cn/en/zhwang), [Meng Li](https://ubiplab.github.io/MengLi_CV.github.io/) & [Wenzhi Chen](http://arc.zju.edu.cn/64010/list.htm) <br> 
>  🎉 *[The paper](src/camera_ready/INFOCOM_2026__Camera_Ready.pdf) has been accepted by [INFOCOM 2026](https://infocom2026.ieee-infocom.org/).*

## 📄 Introduction

In this repo, we implement a novel wide-area-network traffic engineering (WAN TE) framework called ***Lᴍᴛᴇ***, that leverages language models to reason about the WAN TE problem. At a high level, the framework consists of four main components: 1️⃣ domain-aware prompt, 2️⃣ invariant multimodal embedding, 3️⃣ embedding2language alignment, and 4️⃣ lightweight planning head. Lᴍᴛᴇ is the first LM-driven framework that supports effective multimodal alignment and efficient configuration generation both in theory and in practice, while preserving the capabilities of pre-trained LMs. 

### 🎨 Key Features 

- [x] Core implementation of the proposed approach  
- [x] Scripts for fine-tuning and evaluation  
- [x] Baseline implementation support 

## 📁 Repo Structure

```
.
├── cl_baselines/        # Implementation of classical baselines (using gurobi)
├── data/                # Datasets and pre-process code
├── figs/                # Some useful figures
├── lms/                 # Pre-trained language model weights
├── ml_baselines/        # Implementation of learning-based baselines
├── scripts/             # Scripts for running experiments
├── src/                 # Implementation of Lᴍᴛᴇ
├── deepspeed_cfg.json   # DeepSpeed configuration file
├── main.py              # Main entry point
├── README.md            
└── requirements.txt
```

## 📦 Downloads

### 1. Downloading Datasets

In the current release, we only provide a direct link to the *GÉANT* dataset for code testing.
All three real-world traffic matrix datasets can be obtained from the following GitHub repo:

* [Traffic Matrix Prediction Repository](https://github.com/THU-INSC-NAD/Traffic-Matrix-Prediction)

Alternatively, for convenience, we also provide zipped datasets and corresponding topologies via Google Drive:

* [Google Drive Download Link](https://drive.google.com/file/d/1OgvoUrfr_MBLC1Y6ZAS1uX1lvCoz2Cwl/view?usp=sharing)

After downloading, please unzip and place the dataset contents under the `./data` directory.

### 2. Downloading Model Weights

To run this project with **LLaMA 3 series models**, you need to manually download the pretrained model weights from Meta AI and place them under the `./lms` directory.

By default, we use **LLaMA-3-8B** as the backbone large language model.
The official weights are available on [Hugging Face](https://huggingface.co/meta-llama); however, we recommend using [ModelScope](https://www.modelscope.cn/home) for faster and more stable downloads.
Below are example commands for downloading the supported models via ModelScope:

```bash
modelscope download --model LLM-Research/Meta-Llama-3-8B --local_dir Meta-Llama-3-8B
modelscope download --model LLM-Research/Llama-3.2-1B --local_dir Llama-3.2-1B
modelscope download --model LLM-Research/Llama-3.2-3B --local_dir Llama-3.2-3B
modelscope download --model LLM-Research/Meta-Llama-2-13B --local_dir Meta-Llama-2-13B
```

> ⚠️ **Note:** Redistribution of the model weights is prohibited by Meta’s license. Each user must request access individually. Visit the official Meta LLaMA model request page:
   👉 [https://ai.meta.com/resources/models-and-libraries/llama](https://ai.meta.com/resources/models-and-libraries/llama)
   Fill out the request form and wait for approval. Once approved, you will receive download instructions via email. 

## 🚀 Getting Started

We recommend using [Anaconda](https://www.anaconda.com/) to manage dependencies. Create a conda environment with `python=3.10` and activate it. Other versions of python might be okay as well.

```bash
conda create -n env_lmte python==3.10 -y
conda activate env_lmte
```

### 1. Dependencies

Install all required Python packages:

```bash
pip install -r requirements.txt
```

> ⚠️ **Note:** Both the traditional baselines and the optimal solution rely on Gurobi package. Follow [Gurobi Website](https://www.gurobi.com/) to install and setup Gurobi Optimizer. The version used in our paper is **v11.0.3**.

### 2. Running

Once the environment and dependencies are properly set up, you can run Lᴍᴛᴇ on the GÉANT topology using the provided script:

```bash
bash scripts/LMTE_geant.sh
```

## 📜 License

This codebase is licensed under the [Apache License](https://opensource.org/licenses/apache-2-0) - See [LICENSE](https://github.com/Y-debug-sys/LMTE/blob/master/LICENSE) for more details.

## 🔗 Contact

For any technical questions about the paper, please contact [Xinyu Yuan](mailto:yxy5315@gmail.com) (yxy5315@gmail.com) or open an issue on this repository. 

## 🔖 Citation

If you find this work useful for your research, please cite:

```bibtex
@article{yuan2026putting,
  title={LMTE: Putting the ``Reasoning'' into WAN Traffic Engineering with Language Models},
  author={Yuan, Xinyu and Qiao, Yan and Wang, Zonghui and Li, Meng and Chen, Wenzhi},
  journal={arXiv preprint arXiv:2602.00941},
  year={2026}
}
}
```

## ❓ Q & A

1. **Q1:** *Can Lᴍᴛᴇ be applied to other traffic engineering problems?*

   **A1:** Yes. Lᴍᴛᴇ is not limited to minimizing the maximum link utilization (MLU) and can be readily extended to other traffic engineering objectives, such as maximizing total flow throughput (MTF). This extension only requires modifying the loss function during fine-tuning. For implementation details, please refer to the [DOTE codebase](https://github.com/PredWanTE/DOTE) and its [original paper](https://arxiv.org/abs/2303.00735v1).
2. **Q2:** *How can large-scale topologies (e.g., UsCarrier and Cogentco) and their traffic data synthesis be reproduced?*

   **A2:** For each topology, we evaluate Lᴍᴛᴇ using sets of synthetic traffic matrices generated with *gravity* traffic models. These traffic matrices were originally introduced and used in *NCFlow*. The complete set of corresponding traffic matrices and network topologies is publicly available in the NCFlow repository: [https://github.com/netcontract/ncflow](https://github.com/netcontract/ncflow).
3. **Q3:** *Why did my loss become NaN and fail to train the model?*

   **A3:** This is because we randomly introduced failed links during the fine-tuning process, which can lead to a situation where all communication between a pair of nodes is completely interrupted. Although we implemented some safeguards in the selection logic to avoid this, we regret that we cannot fully resolve the issue when a large number (>1) of link failures occur. As an alternative, you can directly disable the option to inject failed links in the configs.

> *<h3>To be continued ...🎬✨</h3>*
