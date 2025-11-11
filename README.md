# Reasoning Transfer for an Extremely Low-resource and Endangered Language: Bridging Languages through Sample-Efficient Language Understanding
*Accepted to AAAI-26*

## Overview
This repository contains code and resources for the paper `Reasoning Alignment for an Extremely Low-resource and Endangered Language: Separating Reasoning and Language Understanding`.

## Structure
- `./data`: evaluation data, including our contributed dataset **LC2024**.
- `./src/train`: training code, including for the baseline Native-CoT Training and **English-Pivoted CoT Training**.
- `./src/evaluation`: evaluation code, based on the [SkyThought](https://github.com/NovaSky-AI/SkyThought) repo.
- `./appendix.pdf`: technical appendix.

## Training
1. Install dependencies:
   ```bash
   cd ./src/train
   pip install -r requirements.txt
   ```

2. Run the main script:
   ```bash
   bash scripts/lang_adapt/run_sft.sh # change the paths accordingly
   ```

## Evaluation
1. Install dependencies:
   ```bash
   cd ./src/evaluation
   pip install -r requirements.txt
   ```

2. Run evaluation:
    ```bash
   cd ./src/evaluation/skythought/skythought_evals
   python eval.py --model ${YOUR_MODEL_HERE}$ --evals=aime,irish_aime,LC2024 --tp=1 --output_file=results.txt --temperatures 0.6 --n 64
   ```


For more information, refer to the original document from the SkyThought repo.

## Citation
TBU
