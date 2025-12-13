# DM-P450: Developing a Multimodal Deep Learning Model for P450 Mining
- [DM-P450: Developing a Multimodal Deep Learning Model for P450 Mining](#dm-p450-developing-a-multimodal-deep-learning-model-for-p450-mining)
- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Citation](#citation)

# Overview
Given the complex interactions between cytochrome P450 enzyme catalytic pockets and their substrates, we developed a multimodal deep learning model (named DM-P450) to predict whether a given P450 enzyme can catalyze a specific substrate molecule.

# Installation

To set up the environment:
``` bash
conda env create -f environment.yml
pip install  dgl -f https://data.dgl.ai/wheels/torch-2.4/cu124/repo.html
pip install boltz[cuda] -U

```
You can download the dataset from **[zenodo.org](https://zenodo.org/records/17861420)** and place it in the **P450_docking/P450_db** folder for use in docking.

# Usage
1. prpare data

 Please place the protein sequences you wish to predict (in FASTA format) and the substrates (in SDF format) into the `P450_docking` folder, and provide them as inputs in the subsequent command-line instructions.

2. Activate the environment
``` bash
conda activate DMP450
```
3. Clean logs and cache (optional)
``` bash
rm -rf ./logs/* DM_P450_model/data/cache/*
```
4. Set up environment variables
``` bash
export PYTHONPATH=$PWD:$PYTHONPATH
```
5. Run inference

Choose the model you want to use from **DM-P450**, **Pocket-P450**, or **Seq-P450**, and provide the corresponding file name as input.
(You only need to input the file name, not the full path, since the files are already placed in the designated directory and the program will automatically locate them.)

When using the DM and Pocket models, the docking process will be involved. After the docking is complete, a prompt will appear on the interface: Please check the modeling results and input the reaction site residue number in uppercase (e.g., C21). You can find the residue number in the output PDB file located at `./P450_docking/*.pdbqt`. Once the site confirmation is completed, the program will automatically proceed with the deep learning prediction task and output the final results.

``` bash
python scripts/infer.py -model DM-P450     -inputFA test.fasta -substrate AGI.sdf | tee logs/infer.log

python scripts/infer.py -model Seq-Only    -inputFA test.fasta -substrate AGI.sdf | tee logs/infer.log

python scripts/infer.py -model Pocket-Only -inputFA test.fasta -substrate AGI.sdf | tee logs/infer.log
```

Or simply run:
``` bash
conda activate DMP450
sh scripts/infer.sh
```
The output results will appear in the `DM_P450_model/data/output` directory and will be stored in CSV format.

The CSV file contains data in the format: `Enzyme_ID, Substrate_ID, Predicted_Probability`, where the maximum probability value is 1.

# Citation

Discovery of Cytochrome P450 Enzymes via Multimodal Deep Learning
