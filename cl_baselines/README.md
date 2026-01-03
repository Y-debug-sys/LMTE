# Preface

The implementations of the classical baselines in this directory are adapted from the official repository of [FIGRET](https://github.com/FIGRET/figret). We sincerely thank the authors for making their code publicly available.

**Which classical algorithms are implemented in this directory?**

1. <u>Linear Programming (LP):</u> A standard traffic engineering formulation that computes optimal routing decisions by directly minimizing congestion under a given traffic demand matrix.

2. <u>Oblivious Routing:</u> Unlike adaptive traffic engineering approaches, oblivious routing optimizes routing configurations to provide worst-case performance guarantees over all possible demand matrices. While this ensures robustness, it often leads to highly sub-optimal performance under typical traffic conditions.

3. <u>COPE:</u> By selecting a worst-case guarantee that is only marginally above the theoretical minimum, COPE optimizes routing based on predicted traffic demands, thereby substantially improving performance in common-case scenarios.

4. <u>Linear Programming (Predicted):</u> This approach first collects a set of historical traffic matrices and then predicts future demands using a weighted moving average. The predicted demand matrix is subsequently used as input to a linear programming formulation to compute the routing configuration.

# How to Run

To run baselines, please try:

```shell
python cl_baselines/run_baseline.py --model optimal       # Linear Programming
python cl_baselines/run_baseline.py --model oblivious     # Oblivious Routing
python cl_baselines/run_baseline.py --model cope          # COPE
python cl_baselines/run_baseline.py --model predte        # Linear Programming (Predicted)
```

For more arguments, please refer to the `run_baseline.py` script.

> ⚠️ **Note:** Before running, please follow [Gurobi Website](https://www.gurobi.com/) to install and setup Gurobi Optimizer. It's free for academic use.

# Citation

If you find our implementations useful, please kindly cite the following papers:

```bibtex
@inproceedings{yuan2026lmte,
  title={LMTE: Putting the ``Reasoning'' into WAN Traffic Engineering with Language Models},
  author={Yuan, Xinyu and Qiao, Yan and Wang, Zonghui and Li, Meng and Chen, Wenzhi},
  booktitle={IEEE INFOCOM 2026-IEEE Conference on Computer Communications},
  year={2026},
  organization={IEEE}
}

@inproceedings{liu2024figret,
  title={Figret: Fine-grained robustness-enhanced traffic engineering},
  author={Liu, Ximeng and Zhao, Shizhen and Cui, Yong and Wang, Xinbing},
  booktitle={Proceedings of the ACM SIGCOMM 2024 Conference},
  pages={117--135},
  year={2024}
}
```
