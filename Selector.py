# Monomer Selection Tool
# Version: 2025.04.01

import os
import torch
import joblib
import numpy as np
import pandas as pd
import itertools
from dataclasses import dataclass
from typing import List, Dict, Tuple
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import Descriptors
from Train import config

# Suppress warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
RDLogger.DisableLog('rdApp.*')


### 1. Data Structures
@dataclass
class MonomerPair:
    """Class for storing monomer pair information and predicted properties."""
    monomer1: str
    monomer2: str
    f1w: float  # Weight fraction of monomer 1
    f1m: float  # Mole fraction of monomer 1
    Tg: float  # Glass transition temperature (K)
    r1: float  # Reactivity ratio 1
    r2: float  # Reactivity ratio 2
    Finst: float  # Instantaneous composition
    kp_avg: float  # Average propagation rate constant
    DaM1: float  # Damkohler number for monomer 1
    DaM2: float  # Damkohler number for monomer 2
    score: float = 0.0  # Overall score


### 2. Model Management
class ModelManager:
    # Class for loading and managing polymer property prediction models
    def __init__(self, models_path: str = "./Train/Files"):
        self.models_path = models_path
        self.models = {}
        self.feature_generators = {}

        # Check if models directory exists
        if not os.path.exists(self.models_path):
            raise FileNotFoundError(f"Models directory not found: {self.models_path}")

        # Define model configurations
        self.model_configs = {
            'Kp': {'model_class': config.ANN_Kp, 'file': 'Trained_Kp1.pt', 'input_dim': config.Full_nbits},
            'Rr': {'model_class': config.ANN_Rr, 'file': 'Trained_Rr1.pt', 'input_dim': 2 * config.Full_nbits},
            'Tg_ANN': {'model_class': config.ANN_Tg, 'file': 'Trained_Tg1.pt', 'input_dim': config.Full_nbits},
            'Tg_CatBoost': {'model_class': config.CATB_Tg, 'file': 'Trained_Tg2_ctb.joblib', 'input_dim': None},
            'Tg_Meta': {'model_class': config.MetaModelANN, 'file': 'Trained_TgO.pt', 'input_dim': None},
            'Ws': {'model_class': config.CATB_Ws, 'file': 'Trained_Ws2_ctb.joblib', 'input_dim': None}}

        # Load all models
        self._load_all_models()

    def _load_all_models(self):
        try:
            # Load ANN models (PyTorch)
            for name in ['Kp', 'Rr', 'Tg_ANN', 'Tg_Meta']:
                self._load_torch_model(name)

            # Load CatBoost models
            for name in ['Tg_CatBoost', 'Ws']:
                self._load_catboost_model(name)

            # Load feature generators
            self._load_feature_generators()

            print(f"\nSuccessfully loaded {len(self.models)} models")

        except Exception as e:
            print(f"Error loading models: {str(e)}")
            raise

    def _load_torch_model(self, model_name: str):
        config_dict = self.model_configs[model_name]
        model_path = os.path.join(self.models_path, config_dict['file'])

        try:
            # Initialize model
            if config_dict['input_dim'] is not None:
                model = config_dict['model_class'](input_dim=config_dict['input_dim'])
            else:
                model = config_dict['model_class']()

            # Load parameters
            checkpoint = torch.load(model_path, weights_only=False, map_location=torch.device('cpu'))
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()

            # Store model and parameters
            self.models[model_name] = {
                'model': model,
                'min_value': checkpoint.get('min_value'),
                'max_value': checkpoint.get('max_value'),
                'feature_params': checkpoint.get('feature_params')}

            print(f"✓ {model_name} model loaded successfully")

        except Exception as e:
            print(f"✗ Failed to load {model_name} model: {str(e)}")
            raise

    def _load_catboost_model(self, model_name: str):
        config_dict = self.model_configs[model_name]
        model_path = os.path.join(self.models_path, config_dict['file'])

        try:
            # Load model
            checkpoint = joblib.load(model_path)

            # Store model and parameters
            self.models[model_name] = {
                'model': checkpoint['model'],
                'min_value': checkpoint.get('min_value'),
                'max_value': checkpoint.get('max_value'),
                'feature_params': checkpoint.get('feature_params')}

            print(f"✓ {model_name} model loaded successfully")

        except Exception as e:
            print(f"✗ Failed to load {model_name} model: {str(e)}")
            raise

    def _load_feature_generators(self):
        try:
            # Load feature generator
            self.feature_generators['Kp'] = config.KpFG.load(self.models['Kp']['feature_params'])
            self.feature_generators['Rr'] = config.RrFG.load(self.models['Rr']['feature_params'])
            self.feature_generators['Tg'] = config.TgFG.load(self.models['Tg_ANN']['feature_params'])
            self.feature_generators['Ws'] = config.WsFG.load(self.models['Ws']['feature_params'])
            print("✓ Feature generators loaded successfully")

        except Exception as e:
            print(f"✗ Failed to load feature generators: {str(e)}")
            raise

    def predict_kp(self, smiles: str) -> float:
        # Generate features
        features = self.feature_generators['Kp'].transform(smiles)

        # Make prediction
        with torch.no_grad():
            X = torch.tensor([features], dtype=torch.float32)
            pred_norm = self.models['Kp']['model'](X).numpy()[0][0]

        # Convert normalized prediction back to original scale and convert from log
        log_kp = config.reverse(pred_norm, self.models['Kp']['min_value'], self.models['Kp']['max_value'])
        predicted_kp = np.exp(log_kp)

        return predicted_kp

    def predict_rr(self, smiles1: str, smiles2: str) -> Tuple[float, float]:
        # Generate features for both monomers
        features1 = self.feature_generators['Rr'].transform(smiles1)
        features2 = self.feature_generators['Rr'].transform(smiles2)

        # Combine features for model input
        X = torch.tensor([np.concatenate([features1, features2])], dtype=torch.float32)

        # Make prediction
        with torch.no_grad():
            pred_norm = self.models['Rr']['model'](X).numpy()[0]

        # Convert normalized predictions back to original scale and convert from log
        log_r1 = config.reverse(pred_norm[0], self.models['Rr']['min_value'][0], self.models['Rr']['max_value'][0])
        log_r2 = config.reverse(pred_norm[1], self.models['Rr']['min_value'][1], self.models['Rr']['max_value'][1])

        r1 = np.exp(log_r1)
        r2 = np.exp(log_r2)

        return r1, r2

    def predict_tg(self, smiles: str) -> float:
        # Generate features
        features = self.feature_generators['Tg'].transform(smiles)

        # Predict using the ANN model
        with torch.no_grad():
            X_ann = torch.tensor([features], dtype=torch.float32)
            ann_pred_norm = self.models['Tg_ANN']['model'](X_ann).item()

        # Predict using the CatBoost model
        X_cat = np.array([features])
        cat_pred_norm = self.models['Tg_CatBoost']['model'].predict(X_cat)[0]

        # Combine predictions for meta-model input
        X_meta = torch.tensor([[ann_pred_norm, cat_pred_norm]], dtype=torch.float32)

        # Predict using the meta-model
        with torch.no_grad():
            meta_pred_norm = self.models['Tg_Meta']['model'](X_meta).item()

        # Convert normalized prediction back to original scale (in Kelvin)
        tg = config.reverse(meta_pred_norm, self.models['Tg_Meta']['min_value'], self.models['Tg_Meta']['max_value']) + 273

        return tg

    def predict_ws(self, smiles: str) -> float:
        # Generate features
        features = self.feature_generators['Ws'].transform(smiles)

        # Make prediction
        X = np.array([features])
        pred_norm = self.models['Ws']['model'].predict(X)[0]

        # Convert normalized prediction back to original scale and convert from log
        log_ws = config.reverse(pred_norm, self.models['Ws']['min_value'], self.models['Ws']['max_value'])
        predicted_ws = np.exp(log_ws)

        return predicted_ws


### 3. Polymer Physics Calculations
class PolymerCalculator:
    @staticmethod
    def weight_to_mole_fraction(f1w: float, smiles1: str, smiles2: str) -> float:
        f2w = 1 - f1w
        mw1 = Descriptors.MolWt(Chem.MolFromSmiles(smiles1))
        mw2 = Descriptors.MolWt(Chem.MolFromSmiles(smiles2))

        n1 = f1w / mw1
        n2 = f2w / mw2
        ntotal = n1 + n2

        return n1 / ntotal

    @staticmethod
    def fox_equation(tg1: float, tg2: float, f1w: float) -> float:
        return 1 / ((f1w / tg1) + ((1 - f1w) / tg2))

    @staticmethod
    def calc_inst_composition(r1: float, r2: float, f1m: float) -> float:
        f2m = 1 - f1m
        return (r1 * f1m ** 2 + f1m * f2m) / (r1 * f1m ** 2 + 2 * f1m * f2m + r2 * f2m ** 2)

    @staticmethod
    def calc_average_kp(kp11: float, kp22: float, r1: float, r2: float, f1m: float) -> float:
        f2m = 1 - f1m
        return (r1 * f1m ** 2 + 2 * f1m * f2m + r2 * f2m ** 2) / ((r1 * f1m / kp11) + (r2 * f2m / kp22))

    @staticmethod
    def calc_damkohler(kp: float, ws: float) -> float:
        # Empirical correlation for emulsion polymerization
        return 1.15e-10 * kp / ws

    @staticmethod
    def correct_water_solubility(ws1: float, ws2: float, f1m: float, f2m: float) -> Tuple[float, float]:
        # Convert to mole fractions in water (55.56 mol/L is concentration of water)
        sol_f1 = ws1 / 55.56  # mole fraction of monomer 1 in water
        sol_f2 = ws2 / 55.56  # mole fraction of monomer 2 in water

        # Correct solubility using co-solvent model (Yalkowsky mixing rule)
        # Note: f1, f2 in the equation are the SOLUBILITY mole fractions, not feed fractions
        ws1_corrected_mf = 10 ** (sol_f2 * np.log10(0.999) + (1 - sol_f2) * np.log10(sol_f1))
        ws1_corrected = ws1_corrected_mf * 55.56

        ws2_corrected_mf = 10 ** (sol_f1 * np.log10(0.999) + (1 - sol_f1) * np.log10(sol_f2))
        ws2_corrected = ws2_corrected_mf * 55.56

        return ws1_corrected, ws2_corrected


### 4. Monomer Selection Algorithm
class MonomerSelector:
    def __init__(self, model_manager: ModelManager, calculator: PolymerCalculator):
        self.model_manager = model_manager
        self.calculator = calculator
        self.monomer_data = None

        # Common tolerance values for selection
        self.tol_composition = 0.3  # Tolerance for composition deviation
        self.tol_kp = 0.2  # Tolerance for kp deviation
        self.tol_damkohler = 0.1  # Tolerance for Damkohler number

    def load_data(self, file_path: str):
        self.monomer_data = pd.read_excel(file_path)
        return self

    def preprocess_data(self):
        if self.monomer_data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        # Generate fingerprints for monomers
        self.monomer_data['FpM'] = [
            self.model_manager.feature_generators['Kp'].transform(smiles)
            for smiles in self.monomer_data['MonSmiles']]

        # Generate fingerprints for polymers
        self.monomer_data['FpP'] = [
            self.model_manager.feature_generators['Tg'].transform(smiles)
            for smiles in self.monomer_data['PolSmiles']]

        # Predict properties for each monomer
        properties = []
        for _, row in self.monomer_data.iterrows():
            props = {}

            # Predict
            props['Kp'] = self.model_manager.predict_kp(row['MonSmiles'])
            props['Tg'] = self.model_manager.predict_tg(row['PolSmiles'])
            props['Ws'] = self.model_manager.predict_ws(row['MonSmiles'])
            properties.append(props)

        # Add properties to dataframe
        for prop in ['Kp', 'Tg', 'Ws']:
            self.monomer_data[prop] = [p[prop] for p in properties]

        return self

    def save_predictions(self, output_file: str):
        if self.monomer_data is None:
            raise ValueError("No data to save. Call preprocess_data() first.")

        # Save to Excel without fingerprint columns (they're large numpy arrays)
        save_cols = [col for col in self.monomer_data.columns if col not in ['FpM', 'FpP']]
        self.monomer_data[save_cols].to_excel(output_file, index=False)

        print(f"Saved predictions to {output_file}")
        return self

    def select_monomers(self, target_pair: Dict, target_tg: float, target_kp: float) -> List[MonomerPair]:
        # Disable criteria to have broader range of candidates
        filtering = False
        # Select and rank monomer pairs based on target properties
        if self.monomer_data is None:
            raise ValueError("No data loaded. Call preprocess_data() first.")

        candidates = []

        # Generate all possible monomer pairs
        monomer_pairs = list(itertools.combinations(range(len(self.monomer_data)), 2))
        print(f"\nEvaluating {len(monomer_pairs)} possible monomer pairs...")

        for idx1, idx2 in monomer_pairs:
            mon1, mon2 = self.monomer_data.iloc[idx1], self.monomer_data.iloc[idx2]

            # Check if Tg values bracket the target
            if not (min(mon1['Tg'], mon2['Tg']) < target_tg < max(mon1['Tg'], mon2['Tg'])):
                continue

            # Calculate required weight fraction to achieve target Tg
            # Solving Fox equation for weight fraction
            fw1 = (((mon1['Tg']) * (mon2['Tg']) / target_tg) - (mon1['Tg'])) / ((mon2['Tg']) - (mon1['Tg']))

            # Calculate predicted Tg with the calculated weight fraction
            tg_copol = self.calculator.fox_equation(mon1['Tg'], mon2['Tg'], fw1)

            # Convert weight fraction to mole fraction
            f1m = self.calculator.weight_to_mole_fraction(fw1, mon1['MonSmiles'], mon2['MonSmiles'])
            f2m = 1 - f1m

            # Calculate reactivity ratios
            r1, r2 = self.model_manager.predict_rr(mon1['MonSmiles'], mon2['MonSmiles'])

            # Calculate instantaneous composition using Mayo-Lewis equation
            Finst = self.calculator.calc_inst_composition(r1, r2, f1m)

            # Check composition drift
            if filtering:
                if abs(Finst - f1m) > self.tol_composition:
                    continue

            # Calculate average kp
            kp_avg = self.calculator.calc_average_kp(mon1['Kp'], mon2['Kp'], r1, r2, f1m)

            # Check polymerization rate
            if filtering:
                if kp_avg < (1 - self.tol_kp) * target_kp:
                    continue

            # Calculate raw Damkohler numbers [in case needed]
            DaM1 = self.calculator.calc_damkohler(kp_avg, mon1['Ws'])
            DaM2 = self.calculator.calc_damkohler(kp_avg, mon2['Ws'])

            # Correct water solubility for co-solvent effect
            ws1_corrected, ws2_corrected = self.calculator.correct_water_solubility(mon1['Ws'], mon2['Ws'], f1m, f2m)

            # Recalculate Damkohler numbers with corrected solubilities
            DaM1_corrected = self.calculator.calc_damkohler(kp_avg, ws1_corrected)
            DaM2_corrected = self.calculator.calc_damkohler(kp_avg, ws2_corrected)

            # Check mass transfer limitations
            if DaM1_corrected > self.tol_damkohler or DaM2_corrected > self.tol_damkohler:
                continue

            # Calculate score based on multiple criteria
            tg_score = 1 - abs(tg_copol - target_tg) / target_tg
            comp_score = 1 - abs(Finst - f1m)
            kp_score = min(kp_avg / target_kp, 1.0)

            # Calculate total score
            total_score = (tg_score + comp_score + kp_score) / 3

            # Add candidate to list
            candidates.append(MonomerPair(
                monomer1=mon1['Monomer'],
                monomer2=mon2['Monomer'],
                f1w=fw1,
                f1m=f1m,
                Tg=tg_copol,
                r1=r1,
                r2=r2,
                Finst=Finst,
                kp_avg=kp_avg,
                DaM1=DaM1_corrected,
                DaM2=DaM2_corrected,
                score=total_score))

        # Sort candidates by score
        candidates.sort(key=lambda x: x.score, reverse=True)

        print(f"Found {len(candidates)} suitable candidates!")
        return candidates


### 5. Common Monomer Database
class CommonMonomers:
    # Database of common monomers
    monomers = {
        "Acrylamide": {"MonSmiles": "NC(=O)C=C", "PolSmiles": "*CC(*)C(=O)N"},
        "Acrylic acid": {"MonSmiles": "OC(=O)C=C", "PolSmiles": "*CC(*)C(=O)O"},
        "Acrylonitrile": {"MonSmiles": "C=CC#N", "PolSmiles": "*CC(*)C#N"},
        "Butadiene": {"MonSmiles": "C=CC=C", "PolSmiles": "*CC=CC*"},
        "Butyl acrylate": {"MonSmiles": "CCCCOC(=O)C=C", "PolSmiles": "*CC(*)C(=O)OCCCC"},
        "Itaconic acid": {"MonSmiles": "OC(=O)CC(=C)C(O)=O", "PolSmiles": "*CC(*)C(=O)O.C(=O)O"},
        "Methyl methacrylate": {"MonSmiles": "COC(=O)C(C)=C", "PolSmiles": "*CC(*)(C)C(=O)OC"},
        "Styrene": {"MonSmiles": "C=Cc1ccccc1", "PolSmiles": "*CC(*)c1ccccc1"},
        "Vinyl acetate": {"MonSmiles": "CC(=O)OC=C", "PolSmiles": "*CC(*)OC(=O)C"},
        "Vinyl chloride": {"MonSmiles": "ClC=C", "PolSmiles": "*CC(*)Cl"},
        "2-Ethylhexyl acrylate": {"MonSmiles": "CCCCC(CC)COC(=O)C=C", "PolSmiles": "*CC(*)C(=O)OCC(CC)CCCC"}}

    # Predefined target copolymer systems
    targets = {
        1: {"name": "MMA/BA 50/50", "M1": "Methyl methacrylate", "M2": "Butyl acrylate", "f1w": 0.5},
        2: {"name": "2-EHA/MMA 90/10", "M1": "2-Ethylhexyl acrylate", "M2": "Methyl methacrylate", "f1w": 0.9},
        3: {"name": "BA/St 90/10", "M1": "Butyl acrylate", "M2": "Styrene", "f1w": 0.9}}

    @classmethod
    def get_target_system(cls, target_id: int) -> Dict:
        # Get monomer data for a specific target system
        if target_id not in cls.targets:
            raise ValueError(f"Target ID {target_id} not found")

        target = cls.targets[target_id]

        return {
            "name": target["name"],
            "M1_MonSmiles": cls.monomers[target["M1"]]["MonSmiles"],
            "M2_MonSmiles": cls.monomers[target["M2"]]["MonSmiles"],
            "M1_PolSmiles": cls.monomers[target["M1"]]["PolSmiles"],
            "M2_PolSmiles": cls.monomers[target["M2"]]["PolSmiles"],
            "f1w": target["f1w"]}


### 6. Main Executable Logic
def main(tid):
    print("Monomer Selection Tool")
    print("=====================")

    # Initialize classes
    model_manager = ModelManager(models_path="./Train/Files")
    calculator = PolymerCalculator()
    selector = MonomerSelector(model_manager, calculator)

    # Load candidate monomers
    selector.load_data("Candidates_inLab.xlsx")

    # Process data and save predictions
    selector.preprocess_data()
    selector.save_predictions("Candidates_calc.xlsx")

    # Select target system
    target_id = tid
    target_system = CommonMonomers.get_target_system(target_id)

    # Calculate target properties
    # Tg prediction for both monomers
    tg1 = model_manager.predict_tg(target_system['M1_PolSmiles'])
    tg2 = model_manager.predict_tg(target_system['M2_PolSmiles'])

    # Convert weight fraction to mole fraction
    f1m = calculator.weight_to_mole_fraction(target_system['f1w'], target_system['M1_MonSmiles'], target_system['M2_MonSmiles'])
    f2m = 1 - f1m

    # Predict reactivity ratios
    r1, r2 = model_manager.predict_rr(target_system['M1_MonSmiles'], target_system['M2_MonSmiles'])

    # Predict Kp values
    kp11 = model_manager.predict_kp(target_system['M1_MonSmiles'])
    kp22 = model_manager.predict_kp(target_system['M2_MonSmiles'])

    # Print selected target system with their properties
    Formula1 = Chem.rdMolDescriptors.CalcMolFormula(Chem.MolFromSmiles(target_system['M1_MonSmiles']))
    Formula2 = Chem.rdMolDescriptors.CalcMolFormula(Chem.MolFromSmiles(target_system['M2_MonSmiles']))
    print(f"\nSelected target system: {target_system['name']}")
    print(f"M1: {Formula1}, kp = {kp11:.1f} L/(mol·s), Tg = {tg1 - 273:.1f}°C, r1 = {r1:.2f}, wt.% = {100 * target_system['f1w']:.1f}")
    print(f"M2: {Formula2}, kp = {kp22:.1f} L/(mol·s), Tg = {tg2 - 273:.1f}°C, r2 = {r2:.2f}, wt.% = {100 * (1 - target_system['f1w']):.1f}")

    # Target Tg using Fox equation
    target_tg = calculator.fox_equation(tg1, tg2, target_system['f1w'])
    # target_tg = 273.15 + -40.15
    # Target Kp using Alfrey equation
    target_kp = calculator.calc_average_kp(kp11, kp22, r1, r2, f1m)

    print(f"\nCalculated target properties:")
    print(f"Target Tg = {target_tg - 273:.1f}°C")
    print(f"Target kp = {target_kp:.1f} L/(mol·s)")

    # Select and rank monomer pairs
    ranked_candidates = selector.select_monomers(target_system, target_tg, target_kp)

    # Print results
    print(f"\nTop 15 candidates for replacing {target_system['name']}:")

    for i, candidate in enumerate(ranked_candidates[:15], 1):
        print(f"\n{i}. {candidate.monomer1}/{candidate.monomer2} ({candidate.f1w:.2f}/{1 - candidate.f1w:.2f})")
        print(f"   Mon1 = {candidate.monomer1}")
        print(f"   Mon2 = {candidate.monomer2}")
        print(f"   Fi = {candidate.Finst:.2f}")
        print(f"   Tg = {candidate.Tg - 273:.1f}°C")
        print(f"   kp = {candidate.kp_avg:.1f} L/(mol·s)")
        print(f"   Dam1 = {candidate.DaM1:.7f}")
        print(f"   Dam2 = {candidate.DaM2:.7f}")
        print(f"   Score: {candidate.score:.3f}")


if __name__ == "__main__":
    # Options 1: MMA/BA 50/50, 2: 2-EHA/MMA 90/10, 3: BA/St 90/10
    tid = 1
    try:
        main(tid)
        print("\nSelection process completed successfully")
    except KeyboardInterrupt:
        print("\nProcess interrupted. Exiting...")
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback

        traceback.print_exc()
