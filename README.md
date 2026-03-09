# redial_eval

This is code repo for ACL-2025 main paper [Assessing Dialect Fairness and Robustness of Large Language Models in Reasoning Tasks](https://aclanthology.org/2025.acl-long.317/).
![ReDial](assets/egs.jpg)

# Website demo
See a demo of our project [here](https://redial-demo.netlify.app/)!

# Direct Use
This is a reasoning dataset created by rewriting 7 existing benchmarks. The rewriting is conducted by human African American Vernacular English (AAVE) speakers, and quality check is conducted by both humans and language models. This could be used for research on testing language models' reasoning fairness and robustness towards AAVE users.

# Out-of-Scope Use
This dataset is being shared for research purposes. For training models to perform real-world tasks, we recommend further testing and validation where needed.
This dataset is not intended for use in educational systems or organizations, or for use in health systems.


You can view data in `data` folder.

# Quick Start
Clone this repo, and install environment by running:
```bash
pip install -r environment.yaml
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
@inproceedings{lin-etal-2025-assessing,
    title = "Assessing Dialect Fairness and Robustness of Large Language Models in Reasoning Tasks",
    author = "Lin, Fangru  and
      Mao, Shaoguang  and
      La Malfa, Emanuele  and
      Hofmann, Valentin  and
      de Wynter, Adrian  and
      Wang, Xun  and
      Chen, Si-Qing  and
      Wooldridge, Michael J.  and
      Pierrehumbert, Janet B.  and
      Wei, Furu",
    editor = "Che, Wanxiang  and
      Nabende, Joyce  and
      Shutova, Ekaterina  and
      Pilehvar, Mohammad Taher",
    booktitle = "Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.acl-long.317/",
    doi = "10.18653/v1/2025.acl-long.317",
    pages = "6317--6342",
    ISBN = "979-8-89176-251-0",
    abstract = "Language is not monolithic. While benchmarks, including those designed for multiple languages, are often used as proxies to evaluate the performance of Large Language Models (LLMs), they tend to overlook the nuances of within-language variation and thus fail to model the experience of speakers of non-standard dialects. Focusing on African American Vernacular English (AAVE), we present the first study aimed at objectively assessing the fairness and robustness of LLMs in handling dialects across canonical reasoning tasks, including algorithm, math, logic, and integrated reasoning. We introduce **ReDial** (**Re**asoning with **Dial**ect Queries), a benchmark containing 1.2K+ parallel query pairs in Standardized English and AAVE. We hire AAVE speakers, including experts with computer science backgrounds, to rewrite seven popular benchmarks,such as HumanEval and GSM8K. With ReDial, we evaluate widely used LLMs, including GPT, Claude, Llama, Mistral, and the Phi model families. Our findings reveal that \textbf{almost all of these widely used models show significant brittleness and unfairness to queries in AAVE}. Our work establishes a systematic and objective framework for analyzing LLM bias in dialectal queries. Moreover, it highlights how mainstream LLMs provide unfair service to dialect speakers in reasoning tasks, laying a critical foundation for future research."
}
```
