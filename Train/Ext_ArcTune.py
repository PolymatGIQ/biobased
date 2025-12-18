# Comprehensive Hyperparameter Tuning
# Version: 2025.04.01

import os
import json
import config
import optuna
import argparse
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import KFold
from typing import Dict, Any, List, Tuple, Optional

import warnings

warnings.filterwarnings('ignore', category=optuna.exceptions.ExperimentalWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="optuna")


### 1. MODEL DEFINITIONS
# 1.1. Dynamic ANN class for Kp1, Tg1, and Ws1 models (single output models)
class DynamicANN_SingleOutput(nn.Module):
    def __init__(self, input_dim: int, layer_sizes: List[int], dropout_rates: Optional[List[float]] = None):
        super(DynamicANN_SingleOutput, self).__init__()
        layers = []
        prev_size = input_dim

        if dropout_rates is None:
            dropout_rates = [0.0] * len(layer_sizes)

        for i, size in enumerate(layer_sizes):
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.ReLU())
            if dropout_rates[i] > 0:
                layers.append(nn.Dropout(dropout_rates[i]))
            prev_size = size

        layers.append(nn.Linear(prev_size, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x).view(-1, 1)


# 1.2. Dynamic ANN class for Kp2 model (dual output)
class DynamicANN_Arrhenius(nn.Module):
    def __init__(self, input_dim: int, layer_sizes: List[int], dropout_rates: Optional[List[float]] = None):
        super(DynamicANN_Arrhenius, self).__init__()
        layers = []
        prev_size = input_dim

        if dropout_rates is None:
            dropout_rates = [0.0] * len(layer_sizes)

        for i, size in enumerate(layer_sizes):
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.ReLU())
            if dropout_rates[i] > 0:
                layers.append(nn.Dropout(dropout_rates[i]))
            prev_size = size

        layers.append(nn.Linear(prev_size, 2))  # Output for Ea and A
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# 1.3. Dynamic ANN class for Rr model (dual output with encoder structure)
class DynamicANN_Rr(nn.Module):
    def __init__(self, input_dim: int, layer_sizes: List[int], dropout_rates: Optional[List[float]] = None,
                 condensed_bits: int = 32):
        super(DynamicANN_Rr, self).__init__()
        self.nbits = input_dim // 2

        # Start with the encoder part (Qe)
        self.Qe = nn.Sequential(
            nn.Linear(self.nbits, layer_sizes[0] if layer_sizes else 64),
            nn.ReLU())

        if len(layer_sizes) > 1:
            self.Qe.add_module("linear2", nn.Linear(layer_sizes[0], condensed_bits))
            self.Qe.add_module("relu2", nn.ReLU())
        else:
            # If only one layer size provided, go directly to condensed_bits
            self.Qe.add_module("linear2", nn.Linear(layer_sizes[0], condensed_bits))
            self.Qe.add_module("relu2", nn.ReLU())

        # Construct the Rr part (taking the concatenated encodings)
        Rr_layers = []
        prev_size = 2 * condensed_bits

        for i, size in enumerate(layer_sizes[1:] if len(layer_sizes) > 1 else [32, 32]):
            Rr_layers.append(nn.Linear(prev_size, size))
            Rr_layers.append(nn.ReLU())
            if dropout_rates and i < len(dropout_rates) and dropout_rates[i] > 0:
                Rr_layers.append(nn.Dropout(dropout_rates[i]))
            prev_size = size

        # Final output layer for the two reactivity ratios
        Rr_layers.append(nn.Linear(prev_size, 2))

        self.Rr = nn.Sequential(*Rr_layers)

    def forward(self, x):
        # Split input into two fingerprints
        M1QE = self.Qe(x[:, :self.nbits])
        M2QE = self.Qe(x[:, self.nbits:])

        # Concatenate the encoder outputs
        xx = torch.cat([M1QE, M2QE], 1)

        # Process through the Rr network
        xx = self.Rr(xx)
        return xx


# 1.4. CatBoost and XGBoost Model Factory Functions
def create_catboost_model(params: Dict[str, Any]):
    from catboost import CatBoostRegressor

    model = CatBoostRegressor(
        iterations=params['iterations'],
        learning_rate=params['learning_rate'],
        depth=params['depth'],
        l2_leaf_reg=params['l2_leaf_reg'],
        loss_function='RMSE',
        random_seed=config.rngch_randstate,
        verbose=False,
        allow_writing_files=False)
    return model


def create_xgboost_model(params: Dict[str, Any]):
    from xgboost import XGBRegressor

    model = XGBRegressor(
        n_estimators=params['n_estimators'],
        learning_rate=params['learning_rate'],
        max_depth=params['max_depth'],
        min_child_weight=params['min_child_weight'],
        subsample=params['subsample'],
        colsample_bytree=params['colsample_bytree'],
        reg_alpha=params['reg_alpha'],
        objective='reg:squarederror',
        random_state=config.rngch_randstate,
        verbosity=0)
    return model


### 2. CONFIGURATION FOR DIFFERENT MODELS
MODEL_CONFIGS = {
    "Kp1": {
        "dataset_path": "./Data/Dataset_Kp.xlsx",
        "target_columns": ["Kp"],
        "feature_generator": config.KpFG,
        "nbits": config.Less_nbits,
        "log_transform": True,
        "model_class": DynamicANN_SingleOutput,
        "default_params": {
            "n_layers": (2, 3),
            "layer_sizes": [(64, 256), (32, 128), (16, 64)],
            "batch_size": (2, 32),
            "k_folds": (5, 10),
            "epochs": (50, 200),
            "learning_rate": (1e-5, 1e-2, "log"),
            "dropout": [(0.0, 0.2), (0.0, 0.1), (0.0, 0.1)]}},

    "Kp2": {
        "dataset_path": "./Data/Dataset_Kp.xlsx",
        "target_columns": ["Ea", "A"],
        "feature_generator": config.KpFG,
        "nbits": config.Less_nbits,
        "log_transform": {"A": True, "Ea": False},
        "model_class": DynamicANN_Arrhenius,
        "default_params": {
            "n_layers": (2, 3),
            "layer_sizes": [(64, 256), (32, 128), (16, 64)],
            "batch_size": (2, 32),
            "k_folds": (5, 10),
            "epochs": (50, 200),
            "learning_rate": (1e-5, 1e-2, "log"),
            "dropout": [(0.0, 0.2), (0.0, 0.1), (0.0, 0.1)]}},

    "Tg1": {
        "dataset_path": "./Data/Dataset_Tg.xlsx",
        "target_columns": ["Tg"],
        "feature_generator": config.TgFG,
        "nbits": config.Less_nbits,
        "log_transform": False,
        "max_smiles_length": 180,
        "model_class": DynamicANN_SingleOutput,
        "default_params": {
            "n_layers": (2, 4),
            "layer_sizes": [(128, 512), (64, 256), (32, 128), (16, 64)],
            "batch_size": (16, 256),
            "k_folds": (5, 10),
            "epochs": (50, 300),
            "learning_rate": (1e-5, 1e-2, "log"),
            "dropout": [(0.0, 0.4), (0.0, 0.3), (0.0, 0.2), (0.0, 0.1)]}},

    "Tg2_ctb": {
        "dataset_path": "./Data/Dataset_Tg.xlsx",
        "target_columns": ["Tg"],
        "feature_generator": config.TgFG,
        "nbits": config.Less_nbits,
        "log_transform": False,
        "max_smiles_length": 180,
        "model_type": "catboost",
        "model_factory": create_catboost_model,
        "default_params": {
            "iterations": (500, 3000),
            "learning_rate": (0.01, 0.1, "log"),
            "depth": (4, 12),
            "l2_leaf_reg": (1, 10),
            "k_folds": (5, 10)}},

    "Tg2_xgb": {
        "dataset_path": "./Data/Dataset_Tg.xlsx",
        "target_columns": ["Tg"],
        "feature_generator": config.TgFG,
        "nbits": config.Less_nbits,
        "log_transform": False,
        "max_smiles_length": 180,
        "model_type": "xgboost",
        "model_factory": create_xgboost_model,
        "default_params": {
            "n_estimators": (500, 3000),
            "learning_rate": (0.01, 0.1, "log"),
            "max_depth": (3, 12),
            "min_child_weight": (1, 10),
            "subsample": (0.6, 1.0),
            "colsample_bytree": (0.6, 1.0),
            "reg_alpha": (0, 10),
            "k_folds": (5, 10)}},

    "Ws1": {
        "dataset_path": "./Data/Dataset_Ws.xlsx",
        "target_columns": ["Ws"],
        "feature_generator": config.WsFG,
        "nbits": config.Less_nbits,
        "log_transform": True,
        "min_value": 1e-10,
        "max_smiles_length": 180,
        "model_class": DynamicANN_SingleOutput,
        "default_params": {
            "n_layers": (2, 3),
            "layer_sizes": [(64, 256), (32, 128), (16, 64)],
            "batch_size": (8, 128),
            "k_folds": (5, 10),
            "epochs": (50, 200),
            "learning_rate": (1e-5, 1e-2, "log"),
            "dropout": [(0.0, 0.2), (0.0, 0.1), (0.0, 0.1)]}},

    "Ws2_ctb": {
        "dataset_path": "./Data/Dataset_Ws.xlsx",
        "target_columns": ["Ws"],
        "feature_generator": config.WsFG,
        "nbits": config.Less_nbits,
        "log_transform": True,
        "min_value": 1e-10,
        "max_smiles_length": 180,
        "model_type": "catboost",
        "model_factory": create_catboost_model,
        "default_params": {
            "iterations": (500, 3000),
            "learning_rate": (0.01, 0.1, "log"),
            "depth": (4, 12),
            "l2_leaf_reg": (1, 10),
            "k_folds": (5, 10)}},

    "Ws2_xgb": {
        "dataset_path": "./Data/Dataset_Ws.xlsx",
        "target_columns": ["Ws"],
        "feature_generator": config.WsFG,
        "nbits": config.Less_nbits,
        "log_transform": True,
        "min_value": 1e-10,
        "max_smiles_length": 180,
        "model_type": "xgboost",
        "model_factory": create_xgboost_model,
        "default_params": {
            "n_estimators": (500, 3000),
            "learning_rate": (0.01, 0.1, "log"),
            "max_depth": (3, 12),
            "min_child_weight": (1, 10),
            "subsample": (0.6, 1.0),
            "colsample_bytree": (0.6, 1.0),
            "reg_alpha": (0, 10),
            "k_folds": (5, 10)}},

    "Rr": {
        "dataset_path": "./Data/Dataset_Rr.xlsx",
        "target_columns": ["r1", "r2"],
        "feature_generator": config.RrFG,
        "nbits": config.Full_nbits,
        "log_transform": True,
        "min_value": 0.002,
        "max_value": 20,
        "model_class": DynamicANN_Rr,
        "is_dual_input": True,
        "input_columns": ["S1", "S2"],
        "default_params": {
            "n_layers": (2, 3),
            "layer_sizes": [(64, 256), (32, 128), (16, 64)],
            "batch_size": (8, 128),
            "k_folds": (5, 10),
            "epochs": (50, 250),
            "learning_rate": (1e-5, 1e-2, "log"),
            "condensed_bits": (16, 64),
            "dropout": [(0.0, 0.3), (0.0, 0.2), (0.0, 0.1)]}}}


### 3. DATA LOADING AND PREPROCESSING
# 3.1 Load and preprocess data for the specified model
def load_data(model_name: str, custom_config: Dict[str, Any] = None) -> Tuple[torch.Tensor, torch.Tensor, Any, Any]:
    # Returns X: Feature tensor, y: Target tensor, and min_value, max_value for normalization
    model_config = MODEL_CONFIGS[model_name].copy()
    if custom_config:
        model_config.update(custom_config)

    # Load the dataset
    df = pd.read_excel(model_config["dataset_path"])

    # Special handling for Rr model with dual inputs
    if model_name == "Rr":
        # Apply filters for Rr
        df = df.loc[(df['r1'] <= model_config["max_value"]) & (df['r1'] >= model_config["min_value"])]
        df = df.loc[(df['r2'] <= model_config["max_value"]) & (df['r2'] >= model_config["min_value"])]
        df = df.reset_index(drop=True)

        # Initialize feature generator
        feature_gen = model_config["feature_generator"](n_bits=model_config["nbits"])

        # Combine both S1 and S2 smiles for fitting
        all_smiles = df['S1'].tolist() + df['S2'].tolist()
        all_fps = feature_gen.fit_transform(all_smiles)

        # Split back into S1 and S2 fingerprints
        n = len(df)
        FP1 = all_fps[:n]
        FP2 = all_fps[n:]

        # Convert to DataFrames and concatenate
        FP1_df = pd.DataFrame(FP1)
        FP2_df = pd.DataFrame(FP2)
        features = pd.concat([FP1_df, FP2_df], axis=1).values

        # Save feature generator
        feature_gen.save('Features_Rr.pkl')

        # Normalize Rr values and combine them
        if model_config["log_transform"]:
            df.r1 = np.log(df.r1)
            df.r2 = np.log(df.r2)

        # Create a 2D array of targets
        Rrs = np.column_stack((df.r1, df.r2))

        # Normalize
        normalized, min_value, max_value = config.normalize(Rrs)

        # Convert to tensors
        X = torch.from_numpy(features).float()
        y = torch.from_numpy(normalized).float()

        return X, y, min_value, max_value

    # For other models - regular flow
    else:
        # Apply filters if any
        if "max_smiles_length" in model_config:
            df = df.loc[df['S'].str.len() <= model_config["max_smiles_length"]]

        if "min_value" in model_config:
            for col in model_config["target_columns"]:
                df = df.loc[df[col] >= model_config["min_value"]]

        # For Kp2, keep only rows with both Ea and A
        if model_name == "Kp2":
            df = df.dropna(subset=model_config["target_columns"])

        # Reset index after filtering
        df = df.reset_index(drop=True)

        # Generate features
        feature_gen = model_config["feature_generator"](n_bits=model_config["nbits"])
        features = feature_gen.fit_transform(df['S'].tolist())

        # Apply log transform if specified
        if model_config["log_transform"]:
            if isinstance(model_config["log_transform"], dict):
                # Apply log transform to specific columns
                for col, apply_log in model_config["log_transform"].items():
                    if apply_log and col in df.columns:
                        df[col] = np.log(df[col])
            else:
                # Apply log transform to all target columns
                for col in model_config["target_columns"]:
                    df[col] = np.log(df[col])

        # Normalize target variables
        if len(model_config["target_columns"]) == 1:
            # Single target variable
            normalized, min_value, max_value = config.normalize(df[model_config["target_columns"][0]].values)
            y = torch.from_numpy(normalized).float().reshape(-1, 1)
        else:
            # Multiple target variables (e.g., Ea and A for Kp2)
            normalized, min_value, max_value = config.normalize(df[model_config["target_columns"]].values)
            y = torch.from_numpy(normalized).float()

        # Convert features to tensor
        X = torch.from_numpy(np.array(features)).float()

        return X, y, min_value, max_value


### 4. OPTUNA OBJECTIVE FUNCTION
# Create an objective function for Optuna to optimize
def create_objective(model_name: str, X: torch.Tensor, y: torch.Tensor,
                     min_value: Any, max_value: Any,
                     param_ranges: Dict[str, Any]) -> callable:
    model_config = MODEL_CONFIGS[model_name]
    is_multi_output = len(model_config["target_columns"]) > 1
    is_Rr = model_name == "Rr"
    is_gbm = "model_type" in model_config and model_config["model_type"] in ["catboost", "xgboost"]

    def objective(trial):
        try:
            # Handle different model types (ANN vs GBM)
            if is_gbm:
                return objective_gbm(trial)
            else:
                return objective_ann(trial)
        except optuna.TrialPruned:
            raise
        except Exception as e:
            print(f"Error in trial {trial.number + 1}: {str(e)}")
            raise

    def objective_ann(trial):
        # First decide number of layers
        n_layers = trial.suggest_int('n_layers', *param_ranges.get('n_layers', (1, 3)))

        # Define layer sizes
        layer_sizes = []
        dropout_rates = []
        for i in range(n_layers):
            if i < len(param_ranges.get('layer_sizes', [])):
                min_size, max_size = param_ranges['layer_sizes'][i]
                layer_sizes.append(trial.suggest_int(f'layer_{i + 1}_size', min_size, max_size, step=16))
            else:
                # Default if not specified
                layer_sizes.append(trial.suggest_int(f'layer_{i + 1}_size', 32, 128, step=16))

            # Add dropout if specified
            if i < len(param_ranges.get('dropout', [])):
                min_rate, max_rate = param_ranges['dropout'][i]
                dropout_rates.append(trial.suggest_float(f'dropout_{i + 1}', min_rate, max_rate, step=0.1))
            else:
                dropout_rates.append(0.0)

        # Training parameters
        batch_size = trial.suggest_int('batch_size', *param_ranges['batch_size'], step=2)
        k_folds = trial.suggest_int('k_folds', *param_ranges['k_folds'], step=5)
        epochs = trial.suggest_int('epochs', *param_ranges['epochs'], step=25)

        # Learning rate can be log-scaled
        lr_config = param_ranges.get('learning_rate', (1e-5, 1e-3, "log"))
        if len(lr_config) == 3 and lr_config[2] == "log":
            learning_rate = trial.suggest_float('learning_rate', lr_config[0], lr_config[1], log=True)
        else:
            learning_rate = trial.suggest_float('learning_rate', lr_config[0], lr_config[1])

        # Special parameter for Rr model - condensed_bits
        condensed_bits = None
        if is_Rr and 'condensed_bits' in param_ranges:
            condensed_bits = trial.suggest_int('condensed_bits', *param_ranges['condensed_bits'], step=8)

        # K-Fold validation
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=config.KFold_randstate)
        val_scores = []

        step_counter = 0  # Single unified counter for all steps

        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            train_dataset = TensorDataset(X_train, y_train)
            val_dataset = TensorDataset(X_val, y_val)

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

            # Create the appropriate model based on the model_name
            if is_Rr:
                model = DynamicANN_Rr(X.shape[1], layer_sizes, dropout_rates,
                                      condensed_bits if condensed_bits else 32)
            elif is_multi_output:
                model = DynamicANN_Arrhenius(X.shape[1], layer_sizes, dropout_rates)
            else:
                model = DynamicANN_SingleOutput(X.shape[1], layer_sizes, dropout_rates)

            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            criterion = nn.MSELoss()

            best_val_loss = float('inf')
            best_model_state = None

            for epoch in range(epochs):
                model.train()
                train_loss = 0
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()

                # Validation
                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        outputs = model(batch_X)
                        val_loss += criterion(outputs, batch_y).item()

                val_loss /= len(val_loader)

                # Report intermediate value every 10 epochs
                if epoch % 10 == 0:
                    trial.report(val_loss, step_counter)
                    step_counter += 1

                    if trial.should_prune():
                        raise optuna.TrialPruned()

                # Store best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = model.state_dict().copy()

            # Calculate metrics for this fold
            model.load_state_dict(best_model_state)
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val)
                mse = criterion(val_pred, y_val).item()
                rmse = np.sqrt(mse)

                # For multi-output models (Kp2 or Rr)
                if is_multi_output:
                    # Calculate specific metrics
                    true_values = config.reverse(y_val.numpy(), min_value, max_value)
                    pred_values = config.reverse(val_pred.numpy(), min_value, max_value)

                    # Special handling for Rr model
                    if is_Rr:
                        # Store metrics for both r1 and r2
                        true_r1, true_r2 = np.exp(true_values[:, 0]), np.exp(true_values[:, 1])
                        pred_r1, pred_r2 = np.exp(pred_values[:, 0]), np.exp(pred_values[:, 1])

                        r1_metrics = config.calc_metrics(true_r1, pred_r1)
                        r2_metrics = config.calc_metrics(true_r2, pred_r2)

                        # Store metrics
                        for metric_name, value in r1_metrics.items():
                            trial.set_user_attr(f'fold_{fold}_r1_{metric_name}', float(value))

                        for metric_name, value in r2_metrics.items():
                            trial.set_user_attr(f'fold_{fold}_r2_{metric_name}', float(value))

                        # Use average RMSE of r1 and r2 as the main score
                        fold_score = (r1_metrics['RMSE'] + r2_metrics['RMSE']) / 2

                    # For Kp2
                    elif model_name == "Kp2":
                        # Store metrics for both Ea and A
                        true_Ea, true_A = true_values[:, 0], np.exp(true_values[:, 1])
                        pred_Ea, pred_A = pred_values[:, 0], np.exp(pred_values[:, 1])

                        ea_metrics = config.calc_metrics(true_Ea, pred_Ea)
                        a_metrics = config.calc_metrics(true_A, pred_A)

                        # Calculate Kp at 60°C (as in Training_Kp2.py)
                        R = 8.314 / 1000  # Gas constant in kJ/(mol·K)
                        T = 273.15 + 60  # Temperature in Kelvin (60°C)

                        true_Kp = true_A * np.exp(-true_Ea / (R * T))
                        pred_Kp = pred_A * np.exp(-pred_Ea / (R * T))

                        kp_metrics = config.calc_metrics(true_Kp, pred_Kp)

                        # Store all metrics
                        for metric_name, value in ea_metrics.items():
                            trial.set_user_attr(f'fold_{fold}_Ea_{metric_name}', float(value))

                        for metric_name, value in a_metrics.items():
                            trial.set_user_attr(f'fold_{fold}_A_{metric_name}', float(value))

                        for metric_name, value in kp_metrics.items():
                            trial.set_user_attr(f'fold_{fold}_Kp_{metric_name}', float(value))

                        # Use RMSE for Kp at 60°C (consistent with Kp1 approach)
                        fold_score = kp_metrics['RMSE']

                else:
                    # Calculate specific metrics for single-output models
                    if model_name in ["Kp1", "Ws1"]:
                        # These have log transforms
                        true_values = np.exp(config.reverse(y_val.numpy(), min_value, max_value))
                        pred_values = np.exp(config.reverse(val_pred.numpy(), min_value, max_value))
                    else:
                        # Tg1 doesn't have log transform
                        true_values = config.reverse(y_val.numpy(), min_value, max_value)
                        pred_values = config.reverse(val_pred.numpy(), min_value, max_value)

                    metrics = config.calc_metrics(true_values, pred_values)

                    # Store metrics
                    for metric_name, value in metrics.items():
                        trial.set_user_attr(f'fold_{fold}_{metric_name}', float(value))

                    fold_score = metrics['RMSE']

                val_scores.append(fold_score)

                # Report fold completion
                trial.report(np.mean(val_scores), step_counter)
                step_counter += 1

            final_score = np.mean(val_scores)
            trial.set_user_attr('final_score', final_score)
            return final_score

    def objective_gbm(trial):
        # Extract model type and factory function
        model_type = model_config["model_type"]
        model_factory = model_config["model_factory"]

        # Generate hyperparameters based on model type
        model_params = {}

        if model_type == "catboost":
            # CatBoost parameters
            if 'iterations' in param_ranges:
                iter_range = param_ranges['iterations']
                model_params['iterations'] = trial.suggest_int('iterations', iter_range[0], iter_range[1], step=200)

            if 'learning_rate' in param_ranges:
                lr_config = param_ranges['learning_rate']
                if len(lr_config) == 3 and lr_config[2] == "log":
                    model_params['learning_rate'] = trial.suggest_float('learning_rate', lr_config[0], lr_config[1], log=True)
                else:
                    model_params['learning_rate'] = trial.suggest_float('learning_rate', lr_config[0], lr_config[1])

            if 'depth' in param_ranges:
                model_params['depth'] = trial.suggest_int('depth', *param_ranges['depth'], step=1)

            if 'l2_leaf_reg' in param_ranges:
                model_params['l2_leaf_reg'] = trial.suggest_int('l2_leaf_reg', *param_ranges['l2_leaf_reg'], step=1)

        elif model_type == "xgboost":
            # XGBoost parameters
            if 'n_estimators' in param_ranges:
                est_range = param_ranges['n_estimators']
                model_params['n_estimators'] = trial.suggest_int('n_estimators', est_range[0], est_range[1], step=200)

            if 'learning_rate' in param_ranges:
                lr_config = param_ranges['learning_rate']
                if len(lr_config) == 3 and lr_config[2] == "log":
                    model_params['learning_rate'] = trial.suggest_float('learning_rate', lr_config[0], lr_config[1], log=True)
                else:
                    model_params['learning_rate'] = trial.suggest_float('learning_rate', lr_config[0], lr_config[1])

            if 'max_depth' in param_ranges:
                model_params['max_depth'] = trial.suggest_int('max_depth', *param_ranges['max_depth'], step=1)

            if 'min_child_weight' in param_ranges:
                model_params['min_child_weight'] = trial.suggest_int('min_child_weight', *param_ranges['min_child_weight'], step=1)

            if 'subsample' in param_ranges:
                sub_range = param_ranges['subsample']
                model_params['subsample'] = trial.suggest_float('subsample', sub_range[0], sub_range[1], step=0.1)

            if 'colsample_bytree' in param_ranges:
                col_range = param_ranges['colsample_bytree']
                model_params['colsample_bytree'] = trial.suggest_float('colsample_bytree', col_range[0], col_range[1], step=0.1)

            if 'reg_alpha' in param_ranges:
                alpha_range = param_ranges['reg_alpha']
                model_params['reg_alpha'] = trial.suggest_float('reg_alpha', alpha_range[0], alpha_range[1], step=1)

        # Get k_folds
        k_folds = trial.suggest_int('k_folds', *param_ranges['k_folds'], step=5)

        # K-Fold validation
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=config.KFold_randstate)
        val_scores = []

        step_counter = 0  # For reporting purposes

        # Convert PyTorch tensors to numpy arrays (for GBMs)
        X_np = X.numpy()
        y_np = y.numpy()

        for fold, (train_idx, val_idx) in enumerate(kf.split(X_np)):
            X_train, X_val = X_np[train_idx], X_np[val_idx]
            y_train, y_val = y_np[train_idx], y_np[val_idx]

            # If y is 2D with single feature, flatten for GBM models
            if y_train.ndim > 1 and y_train.shape[1] == 1:
                y_train = y_train.ravel()
                y_val = y_val.ravel()

            # Create the model with the current parameters
            model = model_factory(model_params)

            # Training
            if model_type == "catboost":
                # CatBoost training
                model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)

            elif model_type == "xgboost":
                # XGBoost training
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

            # Get predictions
            val_pred = model.predict(X_val)

            # Reshape predictions if needed
            if val_pred.ndim == 1:
                val_pred = val_pred.reshape(-1, 1)

            if y_val.ndim == 1:
                y_val = y_val.reshape(-1, 1)

            # Calculate metrics
            if model_name.startswith(("Kp1", "Ws1")) or (model_name.startswith(("Ws2")) and model_config["log_transform"]):
                # These have log transforms
                true_values = np.exp(config.reverse(y_val, min_value, max_value))
                pred_values = np.exp(config.reverse(val_pred, min_value, max_value))
            else:
                # Tg1/Tg2 don't have log transform
                true_values = config.reverse(y_val, min_value, max_value)
                pred_values = config.reverse(val_pred, min_value, max_value)

            metrics = config.calc_metrics(true_values, pred_values)

            # Store metrics
            for metric_name, value in metrics.items():
                trial.set_user_attr(f'fold_{fold}_{metric_name}', float(value))

            fold_score = metrics['RMSE']
            val_scores.append(fold_score)

            # Report fold completion
            trial.report(np.mean(val_scores), step_counter)
            step_counter += 1

        final_score = np.mean(val_scores)
        trial.set_user_attr('final_score', final_score)
        return final_score

    return objective


### 5. HYPERPARAMETER OPTIMIZATION
def optimize_hyperparameters(model_name: str, param_ranges: Dict[str, Any] = None, n_trials: int = 100) -> optuna.study.Study:
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}. Must be one of: {list(MODEL_CONFIGS.keys())}")

    # Set up directory for results
    os.makedirs('./Tuned', exist_ok=True)

    # Suppress verbose Optuna output
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Load data for the selected model
    print(f"Loading data for {model_name}...")
    X, y, min_value, max_value = load_data(model_name)
    print(f"Data loaded. Features shape: {X.shape}, Target shape: {y.shape}")

    # Set parameter ranges, using defaults if not specified
    if param_ranges is None:
        param_ranges = MODEL_CONFIGS[model_name]["default_params"]
    else:
        # Merge with defaults for any missing parameters
        default_params = MODEL_CONFIGS[model_name]["default_params"]
        for param, value in default_params.items():
            if param not in param_ranges:
                param_ranges[param] = value

    # Create the objective function
    objective = create_objective(model_name, X, y, min_value, max_value, param_ranges)

    # Create study with advanced sampling and pruning
    sampler = optuna.samplers.TPESampler(n_startup_trials=10, multivariate=True, warn_independent_sampling=False, seed=42)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10, interval_steps=1, n_min_trials=3)
    study = optuna.create_study(direction='minimize', sampler=sampler, pruner=pruner, study_name=f'{model_name}_optimization')

    # Run the optimization
    print(f"Starting optimization for {model_name} with {n_trials} trials...")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # Get all completed trials
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    # Sort by value (smaller is better since we're minimizing)
    sorted_trials = sorted(completed_trials, key=lambda t: t.value)

    # Save top 10 trials (or fewer if there aren't 10 completed trials)
    top_n = min(10, len(sorted_trials))
    top_trials = []

    for i, t in enumerate(sorted_trials[:top_n]):
        trial_data = {"rank": i + 1, "value": t.value, "params": t.params}
        top_trials.append(trial_data)

    with open(f'./Tuned/{model_name}_Top{top_n}Trials.json', 'w') as f:
        json.dump(top_trials, f, indent=4)

    # Print top 3 trials
    print("\nTop 3 trials:")
    for i, t in enumerate(sorted_trials[:min(3, len(sorted_trials))], 1):
        print(f"\n#{i}: Value: {t.value}")
        print("Parameters:")
        for key, value in t.params.items():
            print(f"  {key}: {value}")

    # Analyze the optimization process
    print("\nStudy statistics:")
    print(f"№ finished trials: {len(study.trials)}")
    print(f"№ pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
    print(f"№ complete trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")

    # Save visualizations
    try:
        # Plot optimization history
        fig = optuna.visualization.plot_optimization_history(study)
        fig.write_html(f"./Tuned/{model_name}_OptCurve.html")

        # Plot parallel coordinate
        fig = optuna.visualization.plot_parallel_coordinate(study)
        fig.write_html(f"./Tuned/{model_name}_ParamRel.html")

        # Plot parameter importance
        fig = optuna.visualization.plot_param_importances(study)
        fig.write_html(f"./Tuned/{model_name}_ParamRank.html")

        # Plot slice
        fig = optuna.visualization.plot_slice(study)
        fig.write_html(f"./Tuned/{model_name}_SlicePlot.html")

    except Exception as e:
        print(f"Warning: Could not create some visualizations: {str(e)}")

    return study


### 6. CUSTOM PARAMETER RANGE CONFIGURATION
# 6.1. handles a single parameter input and validates it
def get_param_range(param_name, default_range, is_int=True, allow_log=False):
    while True:
        try:
            response = input(f"\n{param_name} ({default_range}): ")

            # Handle different input cases
            if not response:
                return default_range  # Use default

            if response.lower() == 'skip':
                return None  # Skip this parameter

            parts = response.split(',')

            # Handle log-scale option for learning rate
            if allow_log and len(parts) == 3 and parts[2].lower() == 'log':
                min_val = float(parts[0])
                max_val = float(parts[1])
                return (min_val, max_val, "log")

            # Ensure we have exactly two values for min and max
            if len(parts) != 2:
                print(f"Error: Please enter two values separated by a comma (min,max).")
                continue

            # Convert to int or float as appropriate
            if is_int:
                min_val = int(parts[0])
                max_val = int(parts[1])
            else:
                min_val = float(parts[0])
                max_val = float(parts[1])

            # Validate min < max
            if min_val >= max_val:
                print(f"Error: Minimum value must be less than maximum value.")
                continue

            return (min_val, max_val)

        except ValueError:
            print(f"Error: Invalid input. Please enter numeric values in the format 'min,max'.")


# 6.2. Interactive function to get custom parameter ranges from the user
def get_param_config_from_user() -> Tuple[str, Dict[str, Any], int]:
    print("\n===== Hyperparameter Optimization Configurator =====")

    # 1. Select model
    print("\nAvailable models:")
    display_models = ["Kp1 [Direct]", "Kp2 [Arrhenius]", "Tg1 [Neural net]", "Tg2 [Gradient boosting]",
                      "Ws1 [Neural net]", "Ws2 [Gradient boosting]", "Rrs [Neural net]"]
    for i, model_display in enumerate(display_models, 1):
        print(f"{i}. {model_display}")

    while True:
        try:
            model_idx = int(input("\nSelect a model (enter number): "))
            if 1 <= model_idx <= len(display_models):
                model_display = display_models[model_idx - 1]

                # Handle special cases
                if model_display == "Tg2":
                    print("\nSelect Tg2 model type:")
                    print("1. CatBoost (Tg2_ctb)")
                    print("2. XGBoost (Tg2_xgb)")
                    sub_choice = input("Enter model type (1-2): ")
                    try:
                        sub_idx = int(sub_choice)
                        if sub_idx == 1:
                            model_name = "Tg2_ctb"
                        elif sub_idx == 2:
                            model_name = "Tg2_xgb"
                        else:
                            print("Invalid choice. Defaulting to CatBoost model.")
                            model_name = "Tg2_ctb"
                    except ValueError:
                        print("Invalid input. Defaulting to CatBoost model.")
                        model_name = "Tg2_ctb"
                elif model_display == "Ws2":
                    print("\nSelect Ws2 model type:")
                    print("1. CatBoost (Ws2_ctb)")
                    print("2. XGBoost (Ws2_xgb)")
                    sub_choice = input("Enter model type (1-2): ")
                    try:
                        sub_idx = int(sub_choice)
                        if sub_idx == 1:
                            model_name = "Ws2_ctb"
                        elif sub_idx == 2:
                            model_name = "Ws2_xgb"
                        else:
                            print("Invalid choice. Defaulting to CatBoost model.")
                            model_name = "Ws2_ctb"
                    except ValueError:
                        print("Invalid input. Defaulting to CatBoost model.")
                        model_name = "Ws2_ctb"
                else:
                    model_name = model_display
                break
            else:
                print("Invalid selection. Please try again.")
        except ValueError:
            print("Please enter a valid number.")

    print(f"\nSelected model: {model_name}")

    # 2. Configure the number of trials
    while True:
        try:
            n_trials = int(input("\nNumber of optimization trials (10-1000, default 100): ") or "100")
            if 10 <= n_trials <= 1000:
                break
            else:
                print("Please enter a value between 10 and 1000.")
        except ValueError:
            print("Please enter a valid number.")

    # 3. Get default parameters for the selected model
    default_params = MODEL_CONFIGS[model_name]["default_params"]

    # 4. Ask if user wants to modify the default parameter ranges
    custom_params = {}
    modify = input("\nDo you want to modify default parameter ranges? (y/n, default: n): ").lower() == 'y'

    if modify:
        print("\nFor each parameter, you can either:")
        print("- Press Enter to keep the default range")
        print("- Enter 'skip' to exclude this parameter from optimization")
        print("- Enter a new range in the format 'min,max")
        print("\nDefault ranges:")

        for param, value in default_params.items():
            print(f"  {param}: {value}")

        # Check if this is a gradient boosting model
        is_gbm = "model_type" in MODEL_CONFIGS[model_name] and MODEL_CONFIGS[model_name]["model_type"] in ["catboost", "xgboost"]

        if is_gbm:
            # Handle gradient boosting specific parameters
            model_type = MODEL_CONFIGS[model_name]["model_type"]

            if model_type == "catboost":
                # CatBoost specific parameters
                iterations_range = get_param_range('iterations', default_params['iterations'])
                if iterations_range is not None:
                    if len(iterations_range) == 2:  # Add step if not provided
                        iterations_range = (iterations_range[0], iterations_range[1], 100)
                    custom_params['iterations'] = iterations_range

                # Learning rate (common to both)
                lr_range = get_param_range('learning_rate', default_params['learning_rate'], is_int=False, allow_log=True)
                if lr_range is not None:
                    custom_params['learning_rate'] = lr_range

                depth_range = get_param_range('depth', default_params['depth'])
                if depth_range is not None:
                    custom_params['depth'] = depth_range

                l2_leaf_reg_range = get_param_range('l2_leaf_reg', default_params['l2_leaf_reg'])
                if l2_leaf_reg_range is not None:
                    custom_params['l2_leaf_reg'] = l2_leaf_reg_range

                # K-folds (last, to match dictionary order)
                k_folds_range = get_param_range('k_folds', default_params['k_folds'])
                if k_folds_range is not None:
                    custom_params['k_folds'] = k_folds_range

            elif model_type == "xgboost":
                # XGBoost specific parameters
                n_estimators_range = get_param_range('n_estimators', default_params['n_estimators'])
                if n_estimators_range is not None:
                    if len(n_estimators_range) == 2:  # Add step if not provided
                        n_estimators_range = (n_estimators_range[0], n_estimators_range[1], 100)
                    custom_params['n_estimators'] = n_estimators_range

                # Learning rate (common to both)
                lr_range = get_param_range('learning_rate', default_params['learning_rate'], is_int=False, allow_log=True)
                if lr_range is not None:
                    custom_params['learning_rate'] = lr_range

                max_depth_range = get_param_range('max_depth', default_params['max_depth'])
                if max_depth_range is not None:
                    custom_params['max_depth'] = max_depth_range

                min_child_weight_range = get_param_range('min_child_weight', default_params['min_child_weight'])
                if min_child_weight_range is not None:
                    custom_params['min_child_weight'] = min_child_weight_range

                subsample_range = get_param_range('subsample', default_params['subsample'], is_int=False)
                if subsample_range is not None:
                    if len(subsample_range) == 2:  # Add step if not provided
                        subsample_range = (subsample_range[0], subsample_range[1], 0.05)
                    custom_params['subsample'] = subsample_range

                colsample_bytree_range = get_param_range('colsample_bytree', default_params['colsample_bytree'], is_int=False)
                if colsample_bytree_range is not None:
                    if len(colsample_bytree_range) == 2:  # Add step if not provided
                        colsample_bytree_range = (colsample_bytree_range[0], colsample_bytree_range[1], 0.05)
                    custom_params['colsample_bytree'] = colsample_bytree_range

                reg_alpha_range = get_param_range('reg_alpha', default_params['reg_alpha'], is_int=False)
                if reg_alpha_range is not None:
                    if len(reg_alpha_range) == 2:  # Add step if not provided
                        reg_alpha_range = (reg_alpha_range[0], reg_alpha_range[1], 1)
                    custom_params['reg_alpha'] = reg_alpha_range

                # K-folds (last, to match dictionary order)
                k_folds_range = get_param_range('k_folds', default_params['k_folds'])
                if k_folds_range is not None:
                    custom_params['k_folds'] = k_folds_range

        else:
            # Process each parameter type using the get_param_range helper for ANN models
            # Number of layers
            n_layers_range = get_param_range('n_layers', default_params['n_layers'])
            if n_layers_range is not None:
                custom_params['n_layers'] = n_layers_range

            # Layer sizes
            custom_layer_sizes = []
            for i in range(len(default_params['layer_sizes'])):
                default_range = default_params['layer_sizes'][i]
                layer_range = get_param_range(f'layer_{i + 1}_size', default_range)

                if layer_range is None:
                    # User wants to skip this and remaining layers
                    if i == 0:
                        # Can't skip the first layer
                        custom_layer_sizes.append(default_range)
                    break
                else:
                    custom_layer_sizes.append(layer_range)

            if custom_layer_sizes:
                custom_params['layer_sizes'] = custom_layer_sizes

            # Batch size
            batch_size_range = get_param_range('batch_size', default_params['batch_size'])
            if batch_size_range is not None:
                custom_params['batch_size'] = batch_size_range

            # K-folds
            k_folds_range = get_param_range('k_folds', default_params['k_folds'])
            if k_folds_range is not None:
                custom_params['k_folds'] = k_folds_range

            # Epochs
            epochs_range = get_param_range('epochs', default_params['epochs'])
            if epochs_range is not None:
                custom_params['epochs'] = epochs_range

            # Learning rate
            lr_range = get_param_range('learning_rate', default_params['learning_rate'], is_int=False, allow_log=True)
            if lr_range is not None:
                custom_params['learning_rate'] = lr_range

            # Special parameter for Rr model - condensed_bits
            if model_name == "Rr" and 'condensed_bits' in default_params:
                condensed_bits_range = get_param_range('condensed_bits', default_params['condensed_bits'])
                if condensed_bits_range is not None:
                    custom_params['condensed_bits'] = condensed_bits_range

            # Dropout rates
            custom_dropout = []
            for i in range(len(default_params.get('dropout', []))):
                default_range = default_params['dropout'][i]
                dropout_range = get_param_range(f'dropout_{i + 1}', default_range, is_int=False)

                if dropout_range is None:
                    break
                else:
                    custom_dropout.append(dropout_range)

            if custom_dropout:
                custom_params['dropout'] = custom_dropout

    # If custom_params is empty, use default_params
    param_ranges = custom_params if custom_params else default_params

    # Print final configuration
    print("\n===== Final Configuration =====")
    print(f"Model: {model_name}")
    print(f"Number of trials: {n_trials}")
    print("Parameter ranges:")

    for param, value in param_ranges.items():
        print(f"  {param}: {value}")
    while True:
        confirm = input("\nProceed with this configuration? (y/n): ").lower()
        if confirm == 'y':
            break
        elif confirm == 'n':
            print("\nRestarting configuration...\n")
            return get_param_config_from_user()  # Recursively restart
        else:
            print("Please enter 'y' to proceed or 'n' to restart.")

    return model_name, param_ranges, n_trials


### 7. MAIN FUNCTION
def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Hyperparameter tuning for polymer property prediction models')
    parser.add_argument('--model', type=str, choices=list(MODEL_CONFIGS.keys()), help='Model to optimize (Kp1, Kp2, Tg1, Tg2_ctb, Tg2_xgb, Ws1, Ws2_ctb, Ws2_xgb, Rr)')
    parser.add_argument('--trials', type=int, default=100, help='Number of optimization trials to run (default: 100)')
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode to customize parameter ranges')
    parser.add_argument('--param_file', type=str, help='JSON file with custom parameter ranges')
    args = parser.parse_args()

    # Determine which mode to run in
    if args.interactive:
        # Interactive mode: get configuration from user
        model_name, param_ranges, n_trials = get_param_config_from_user()
        study = optimize_hyperparameters(model_name, param_ranges, n_trials)

    elif args.model:
        # Command line mode with specified model
        model_name = args.model
        param_ranges = None
        n_trials = args.trials

        # Load custom parameter ranges from file if specified
        if args.param_file:
            try:
                with open(args.param_file, 'r') as f:
                    param_ranges = json.load(f)
                print(f"Loaded custom parameter ranges from {args.param_file}")
            except Exception as e:
                print(f"Error loading parameter file: {str(e)}")
                print("Using default parameter ranges instead.")

        study = optimize_hyperparameters(model_name, param_ranges, n_trials)

    else:
        # No model specified, show help and run interactive mode
        parser.print_help()
        print("\nNo model specified. Starting interactive mode...\n")
        model_name, param_ranges, n_trials = get_param_config_from_user()
        study = optimize_hyperparameters(model_name, param_ranges, n_trials)

    print("\nOptimization complete!")
    print(f"Best parameters saved")
    print("Visualization plots saved")


if __name__ == "__main__":
    main()

