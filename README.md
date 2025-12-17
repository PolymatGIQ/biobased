# Bio-Based Monomer Selection Tool

This repository contains code for **selecting bio-based monomers to replace petroleum-based monomers in emulsion polymerization**. We developed machine learning models that predict key polymer properties and use them to find sustainable alternatives that match the performance of conventional systems. More details are described in **our paper: [Title] [DOI/Link - to be added after publication]**.

## What This Project Does

This tool helps researchers find bio-based replacements for conventional monomers used in polymer production. It uses machine learning to predict four important properties:
- **Propagation rate constant (Kp)**
- **Reactivity ratios (Rr)** 
- **Glass transition temperature (Tg)**
- **Water solubility (Ws)**

The tool then finds pairs of bio-based monomers that can replace conventional ones while keeping similar properties.

## Repository Structure

```
biobased/
├── Selector.py                    # Main selection tool
├── Candidates_inLab.xlsx          # List of candidate bio-based monomers
├── Train/                         # Training scripts and models
│   ├── config.py                  # Configuration and model definitions
│   ├── Training_Kp.py            # Train Kp prediction model
│   ├── Training_Rr.py            # Train reactivity ratios model
│   ├── Training_Tg1.py           # Train Tg neural network model
│   ├── Training_Tg2.py           # Train Tg gradient boosting model
│   ├── Training_TgT.py           # Train Tg meta-model
│   ├── Training_Ws.py            # Train water solubility model
│   ├── Ext_Analysis.py           # Analyze molecular descriptors
│   └── Files/                    # Trained model files
│       ├── Trained_Kp1.pt
│       ├── Trained_Rr1.pt
│       ├── Trained_Tg1.pt
│       ├── Trained_Tg2_ctb.joblib
│       ├── Trained_TgO.pt
│       └── Trained_Ws2_ctb.joblib
```

## File Descriptions

### Main Files

**`Selector.py`** - The main selection tool
- **What it does:** Takes a target monomer pair (like MMA/BA 50/50) and finds bio-based alternatives
- **Uses:** All trained models in `Train/Files/` and candidate list in `Candidates_inLab.xlsx`
- **Creates:** `Candidates_calc.xlsx` (predicted properties for all candidates)
- **Shows:** Top 15 bio-based monomer pairs ranked by how well they match the target

**`Candidates_inLab.xlsx`** - Input file
- **Contains:** List of bio-based monomers to consider as replacements
- **Columns:** Monomer name, SMILES notation for monomer and polymer structures

### Training Scripts

**`config.py`** - Configuration file
- **What it does:** Defines all model architectures, feature generators, and helper functions
- **Contains:** Neural network models, gradient boosting models, normalization functions, evaluation metrics
- **Used by:** All other training scripts

**`Training_Kp.py`** - Train propagation rate model
- **Uses:** `Dataset_Kp.xlsx`
- **Creates:** `Trained_Kp1.pt` in `Train/Files/`
- **Output:** Model that predicts how fast polymerization happens

**`Training_Rr.py`** - Train reactivity ratios model
- **Uses:** `Dataset_Rr.xlsx`
- **Creates:** `Trained_Rr1.pt` in `Train/Files/`
- **Output:** Model that predicts reactivity ratios for monomer pairs

**`Training_Tg1.py`** - Train glass transition temperature (neural network)
- **Uses:** `Dataset_Tg.xlsx` (from Data folder, not included in repo)
- **Creates:** `Trained_Tg1.pt` in `Train/Files/`
- **Output:** First Tg prediction model (ANN approach)

**`Training_Tg2.py`** - Train glass transition temperature (gradient boosting)
- **Uses:** `Dataset_Tg.xlsx`
- **Creates:** `Trained_Tg2_ctb.joblib` in `Train/Files/`
- **Output:** Second Tg prediction model (CatBoost approach)

**`Training_TgT.py`** - Train glass transition temperature (meta-model)
- **Uses:** Both `Trained_Tg1.pt` and `Trained_Tg2_ctb.joblib`
- **Creates:** `Trained_TgO.pt` in `Train/Files/`
- **Output:** Final Tg model that combines the two previous models for better accuracy

**`Training_Ws.py`** - Train water solubility model
- **Uses:** `Dataset_Ws.xlsx`
- **Creates:** `Trained_Ws2_ctb.joblib` in `Train/Files/`
- **Output:** Model that predicts water solubility of monomers

**`Ext_Analysis.py`** - Analyze molecular descriptors
- **What it does:** Helps identify which molecular features are most important for each property
- **Uses:** Any dataset file you provide
- **Creates:** Correlation heatmaps and descriptor rankings in `Analyze/` folder
- **Purpose:** Used during model development to select best descriptors

### Trained Models

The `Train/Files/` folder contains six trained models:
- `Trained_Kp1.pt` - Propagation rate constant model
- `Trained_Rr1.pt` - Reactivity ratios model
- `Trained_Tg1.pt` - Glass transition temperature (neural network)
- `Trained_Tg2_ctb.joblib` - Glass transition temperature (gradient boosting)
- `Trained_TgO.pt` - Glass transition temperature (meta-model)
- `Trained_Ws2_ctb.joblib` - Water solubility model

## How to Use

### Quick Start

1. **Install required packages:**
```bash
pip install torch pandas numpy rdkit scikit-learn catboost joblib openpyxl
```

2. **Run the selector:**
```bash
python Selector.py
```

The tool will:
- Load all trained models
- Read the candidate monomers from `Candidates_inLab.xlsx`
- Calculate properties for each candidate
- Find the best bio-based replacements for the target system
- Show the top 15 candidates with their predicted properties

### Customize Target System

Open `Selector.py` and change the target system at the bottom of the file:
```python
# Choose target system: 1, 2, or 3
tid = 1  # 1: MMA/BA 50/50, 2: 2-EHA/MMA 90/10, 3: BA/St 90/10
```

### Add Your Own Candidates

Edit `Candidates_inLab.xlsx` to add or remove candidate monomers. You need:
- Monomer name
- Monomer SMILES notation
- Polymer SMILES notation

## Requirements

- Python 3.8 or higher
- PyTorch
- pandas
- numpy
- RDKit
- scikit-learn
- CatBoost
- joblib
- openpyxl

## Training Your Own Models

If you want to retrain models with your own data:

1. Place your datasets in a `Data/` folder
2. Run the training scripts in order:
   - `Training_Kp.py`
   - `Training_Rr.py`
   - `Training_Tg1.py`
   - `Training_Tg2.py` (after Tg1)
   - `Training_TgT.py` (after Tg1 and Tg2)
   - `Training_Ws.py`

Each script will save the trained model in `Train/Files/`.


## Citation

If you use this code in your research, please cite:

```
[Citation format to be added after publication]
```

## License
This repository is licensed under CC BY-NC 4.0.
For more information please refer to the [license section](https://github.com/PolymatGIQ/biobased/blob/main/License.md).


