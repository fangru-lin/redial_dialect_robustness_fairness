# redial_eval

This is code repo for [One Language, Many Gaps: Evaluating Dialect Fairness and Robustness of Large Language Models in Reasoning Tasks](https://arxiv.org/abs/2410.11005v1).
![ReDial](assets/‎annotation.png)

# Direct Use
This is a reasoning dataset created by rewriting 7 existing benchmarks. The rewriting is conducted by human African American Vernacular English (AAVE) speakers, and quality check is conducted by both humans and language models. This could be used for research on testing language models' reasoning fairness and robustness towards AAVE users.

# Out-of-Scope Use
This dataset is being shared for research purposes. For training models to perform real-world tasks, we recommend further testing and validation where needed.
This dataset is not intended for use in educational systems or organizations, or for use in health systems.


You can view data in `data` folder.

# Quick Start
Clone this repo, and install environment by running:
```bash
pip install -r requirements.txt
```

Activate the environment, then you can run the following command to evaluate on the dataset:
```
bash py_scripts/run_eval.sh
```

For ablation studies, run this command:
```
bash py_scripts/run_ablation.sh
```

For evaluation of algorithm, go to ```scripts``` folder and follow the instructions in clean_and_test_code.ipynb. For other evaluations, you should be able to see the evaluation results right after you get the generation.

# Citation
If you use this dataset in your research, please cite the following paper:
```
@article{lin2024one,
  title={One Language, Many Gaps: Evaluating Dialect Fairness and Robustness of Large Language Models in Reasoning Tasks},
  author={Lin, Fangru and Mao, Shaoguang and La Malfa, Emanuele and Hofmann, Valentin and de Wynter, Adrian and Yao, Jing and Chen, Si-Qing and Wooldridge, Michael and Wei, Furu},
  journal={arXiv preprint arXiv:2410.11005},
  year={2024}
}
```