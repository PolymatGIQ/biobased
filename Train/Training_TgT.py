# Meta-Model for Tg Prediction
# Version: 2025.10.22

import os
import config
import numpy as np
import pandas as pd
import joblib
import torch
from tabulate import tabulate
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Select the meta-model mode
mode = 1  # Change this value to select the desired meta-model
# mode = 1: Original meta-model (simple 2-input from our trained Tg models)
# mode = 2: Enhanced meta-model (compresses full feature vector + 2 inputs)
# mode = 3: Combined meta-model (separate compresses fp and dsc + 2 inputs)


### 1. MODEL ARCHITECTURES
# 1.1. Original Meta-Model (mode = 1)
# Already defined in config.py as MetaModelANN

# 1.2. Enhanced Meta-Model (mode = 2)
class EnhancedMetaModelANN(torch.nn.Module):
    def __init__(self, input_dim):
        super(EnhancedMetaModelANN, self).__init__()
        # Feature compression network
        self.feature_compressor = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, 14))

        # Meta-model combining compressed features and model predictions
        self.meta_model = torch.nn.Sequential(
            torch.nn.Linear(16, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1))

    def forward(self, x_features, x_predictions):
        # Compress molecular features
        compressed_features = self.feature_compressor(x_features)

        # Concatenate compressed features with model predictions
        combined = torch.cat([compressed_features, x_predictions], dim=1)

        # Pass through meta-model
        return self.meta_model(combined)


# 1.3. Combined Meta-Model (mode = 3)
class CombinedMetaModelANN(torch.nn.Module):
    def __init__(self, n_fingerprints, n_descriptors):
        super(CombinedMetaModelANN, self).__init__()
        # Fingerprint compression network
        self.fingerprint_compressor = torch.nn.Sequential(
            torch.nn.Linear(n_fingerprints, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(64, 20))

        # Descriptor compression network
        self.descriptor_compressor = torch.nn.Sequential(
            torch.nn.Linear(n_descriptors, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 6))

        # Meta-model combining all features and model predictions
        self.meta_model = torch.nn.Sequential(
            torch.nn.Linear(28, 32),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(32, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 1))

    def forward(self, x_fingerprints, x_descriptors, x_predictions):
        # Compress fingerprint features
        compressed_fingerprints = self.fingerprint_compressor(x_fingerprints)

        # Compress descriptor features
        compressed_descriptors = self.descriptor_compressor(x_descriptors)

        # Concatenate all features with model predictions
        combined = torch.cat([compressed_fingerprints, compressed_descriptors, x_predictions], dim=1)

        # Pass through meta-model
        return self.meta_model(combined)


### 2. MAIN TRAINING SCRIPT
# Training parameters
n_epochs = 2000
patience = 200
min_delta = 1e-5

# Load base models
print("Loading base models...")
ann_checkpoint = torch.load('./Files/Trained_Tg1.pt', weights_only=False)
ann_model = config.ANN_Tg(input_dim=ann_checkpoint['input_dim'])
ann_model.load_state_dict(ann_checkpoint['model_state_dict'])
ann_model.eval()

catboost_checkpoint = joblib.load('./Files/Trained_Tg2_ctb.joblib')
catboost_model = catboost_checkpoint['model']

# Get normalization values from models
ann_min_value = ann_checkpoint['min_value']
ann_max_value = ann_checkpoint['max_value']
cat_min_value = catboost_checkpoint['min_value']
cat_max_value = catboost_checkpoint['max_value']

# Ensure normalization values are the same
if ann_min_value != cat_min_value or ann_max_value != cat_max_value:
    print(f"Warning: Normalization values differ between models!")
    print(f"ANN: min={ann_min_value}, max={ann_max_value}")
    print(f"CatBoost: min={cat_min_value}, max={cat_max_value}")

# Using the same normalization values for meta-model
meta_min_value = ann_min_value
meta_max_value = ann_max_value

# Importing dataset
print("Loading dataset...")
df = pd.read_excel("./Data/Dataset_Tg.xlsx")
df = df.loc[df['S'].str.len() <= 180]
df = df.reset_index(drop=True)

# Load feature generator
feature_gen = config.TgFG.load(ann_checkpoint.get('feature_params'))
features = [feature_gen.transform(smiles) for smiles in df['S'].tolist()]
X = np.array(features)

# For mode 3, separate fingerprints and descriptors
if mode == 3:
    n_bits = feature_gen.n_bits
    n_descriptors = len(feature_gen.descriptors)
    print(f"Feature dimensions: {n_bits} fingerprint bits + {n_descriptors} descriptors = {X.shape[1]} total")

    # Extract fingerprint and descriptor parts separately
    fingerprint_indices = slice(0, n_bits)
    descriptor_indices = slice(n_bits, n_bits + n_descriptors)

    X_fingerprints = X[:, fingerprint_indices]
    X_descriptors = X[:, descriptor_indices]

    print(f"Fingerprint shape: {X_fingerprints.shape}, Descriptor shape: {X_descriptors.shape}")

# Normalizing Tg
normalized = config.normalize(df['Tg'].values)
y_norm, _, _ = normalized  # We already have min/max from the models

# Split dataset into train and validation
if mode == 3:
    X_train, X_val, X_fp_train, X_fp_val, X_desc_train, X_desc_val, y_train, y_val = train_test_split(
        X, X_fingerprints, X_descriptors, y_norm, test_size=0.1, random_state=config.KFold_randstate)
else:
    X_train, X_val, y_train, y_val = train_test_split(X, y_norm, test_size=0.1, random_state=config.KFold_randstate)

print(f"Training data shape: {X_train.shape}, Validation data shape: {X_val.shape}")

# Generate predictions from base models
print("Generating base model predictions...")

# ANN Predictions
with torch.no_grad():
    X_train_tensor = torch.from_numpy(X_train).float()
    X_val_tensor = torch.from_numpy(X_val).float()

    ann_train_preds = ann_model(X_train_tensor).numpy()
    ann_val_preds = ann_model(X_val_tensor).numpy()

# CatBoost Predictions
catboost_train_preds = catboost_model.predict(X_train).reshape(-1, 1)
catboost_val_preds = catboost_model.predict(X_val).reshape(-1, 1)

# Combine predictions for meta-model
meta_train_X_preds = np.hstack([ann_train_preds, catboost_train_preds])
meta_val_X_preds = np.hstack([ann_val_preds, catboost_val_preds])

# Convert to PyTorch tensors
meta_train_X_preds_tensor = torch.tensor(meta_train_X_preds, dtype=torch.float32)
meta_val_X_preds_tensor = torch.tensor(meta_val_X_preds, dtype=torch.float32)

# Additional tensors based on mode
if mode >= 2:
    # Feature tensors for mode 2 and 3
    meta_train_X_features_tensor = torch.tensor(X_train, dtype=torch.float32)
    meta_val_X_features_tensor = torch.tensor(X_val, dtype=torch.float32)

if mode == 3:
    # Feature tensors specifically for mode 3
    meta_train_X_fp_tensor = torch.tensor(X_fp_train, dtype=torch.float32)
    meta_val_X_fp_tensor = torch.tensor(X_fp_val, dtype=torch.float32)
    meta_train_X_desc_tensor = torch.tensor(X_desc_train, dtype=torch.float32)
    meta_val_X_desc_tensor = torch.tensor(X_desc_val, dtype=torch.float32)

y_train_tensor = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).reshape(-1, 1)

# Initialize the appropriate meta-model based on mode
if mode == 1:
    model_name = "Original"
    meta_model = config.MetaModelANN()
    learning_rate = 1e-3
    weight_decay = 0

elif mode == 2:
    model_name = "Enhanced"
    input_dim = X_train.shape[1]
    meta_model = EnhancedMetaModelANN(input_dim)
    learning_rate = 8e-4
    weight_decay = 0

elif mode == 3:
    model_name = "Combined"
    meta_model = CombinedMetaModelANN(n_fingerprints=n_bits, n_descriptors=n_descriptors)
    learning_rate = 2e-4
    weight_decay = 1e-5

else:
    raise ValueError("Invalid mode. Please select 1, 2, or 3.")

print(f"Training {model_name} meta-model...")

# Training setup
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(meta_model.parameters(), lr=learning_rate, weight_decay=weight_decay)

best_val_loss = float('inf')
best_model_state = None
epochs_without_improvement = 0

# Training loop
for epoch in range(n_epochs):
    # Train mode
    meta_model.train()
    optimizer.zero_grad()

    # Forward pass based on model type
    if mode == 1:
        train_predictions = meta_model(meta_train_X_preds_tensor)
    elif mode == 2:
        train_predictions = meta_model(meta_train_X_features_tensor, meta_train_X_preds_tensor)
    elif mode == 3:
        train_predictions = meta_model(meta_train_X_fp_tensor, meta_train_X_desc_tensor, meta_train_X_preds_tensor)

    train_loss = criterion(train_predictions, y_train_tensor)
    train_loss.backward()
    optimizer.step()

    # Validation mode
    meta_model.eval()
    with torch.no_grad():
        # Forward pass based on model type
        if mode == 1:
            val_predictions = meta_model(meta_val_X_preds_tensor)
        elif mode == 2:
            val_predictions = meta_model(meta_val_X_features_tensor, meta_val_X_preds_tensor)
        elif mode == 3:
            val_predictions = meta_model(meta_val_X_fp_tensor, meta_val_X_desc_tensor, meta_val_X_preds_tensor)

        val_loss = criterion(val_predictions, y_val_tensor)

    # Print progress
    if (epoch + 1) % 50 == 0 or epoch == 0:
        print(f"Epoch {epoch + 1}/{n_epochs} - Train Loss: {train_loss.item():.5f} - Val Loss: {val_loss.item():.5f}")

    # Check if this is the best model
    if val_loss < best_val_loss - min_delta:
        best_val_loss = val_loss.item()
        best_model_state = meta_model.state_dict().copy()
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    # Early stopping
    if epochs_without_improvement >= patience:
        print(f"Early stopping triggered after {epoch + 1} epochs")
        break

# Load best model state
if best_model_state is not None:
    meta_model.load_state_dict(best_model_state)
meta_model.eval()

# Save Meta-Model with appropriate filename based on mode
os.makedirs("./Files", exist_ok=True)
save_dict = {
    'model_state_dict': meta_model.state_dict(),
    'min_value': meta_min_value,
    'max_value': meta_max_value,
    'feature_params': ann_checkpoint['feature_params']}

if mode == 1:
    save_dict['input_dim'] = 2
    filename = 'Trained_TgO.pt'
elif mode == 2:
    save_dict['input_dim'] = X_train.shape[1]
    filename = 'Trained_TgE.pt'
elif mode == 3:
    save_dict['n_fingerprints'] = n_bits
    save_dict['n_descriptors'] = n_descriptors
    filename = 'Trained_TgC.pt'

torch.save(save_dict, f'./Files/{filename}')

# Make predictions for metrics calculation
with torch.no_grad():
    # Get normalized predictions for the selected meta-model
    if mode == 1:
        meta_train_preds = meta_model(meta_train_X_preds_tensor).numpy().flatten()
        meta_val_preds = meta_model(meta_val_X_preds_tensor).numpy().flatten()
    elif mode == 2:
        meta_train_preds = meta_model(meta_train_X_features_tensor, meta_train_X_preds_tensor).numpy().flatten()
        meta_val_preds = meta_model(meta_val_X_features_tensor, meta_val_X_preds_tensor).numpy().flatten()
    elif mode == 3:
        meta_train_preds = meta_model(meta_train_X_fp_tensor, meta_train_X_desc_tensor, meta_train_X_preds_tensor).numpy().flatten()
        meta_val_preds = meta_model(meta_val_X_fp_tensor, meta_val_X_desc_tensor, meta_val_X_preds_tensor).numpy().flatten()

    # Reverse normalization for all predictions
    true_train_values = config.reverse(y_train, meta_min_value, meta_max_value)
    true_val_values = config.reverse(y_val, meta_min_value, meta_max_value)

    # Base models
    ann_train_values = config.reverse(ann_train_preds.flatten(), meta_min_value, meta_max_value)
    cat_train_values = config.reverse(catboost_train_preds.flatten(), meta_min_value, meta_max_value)

    ann_val_values = config.reverse(ann_val_preds.flatten(), meta_min_value, meta_max_value)
    cat_val_values = config.reverse(catboost_val_preds.flatten(), meta_min_value, meta_max_value)

    # Meta-model
    meta_train_values = config.reverse(meta_train_preds, meta_min_value, meta_max_value)
    meta_val_values = config.reverse(meta_val_preds, meta_min_value, meta_max_value)

# Calculate metrics for all models
ann_train_metrics = config.calc_metrics(true_train_values, ann_train_values)
ann_val_metrics = config.calc_metrics(true_val_values, ann_val_values)

cat_train_metrics = config.calc_metrics(true_train_values, cat_train_values)
cat_val_metrics = config.calc_metrics(true_val_values, cat_val_values)

meta_train_metrics = config.calc_metrics(true_train_values, meta_train_values)
meta_val_metrics = config.calc_metrics(true_val_values, meta_val_values)

# Display metrics
print("\nANN Model Metrics:")
config.disp_metrics(ann_train_metrics, ann_val_metrics, "ANN model Tg")

print("\nCatBoost Model Metrics:")
config.disp_metrics(cat_train_metrics, cat_val_metrics, "CatBoost model Tg")

print(f"\n{model_name} Meta-Model Metrics:")
config.disp_metrics(meta_train_metrics, meta_val_metrics, f"{model_name} Meta-Model Tg")

# First combine all true and predicted values
true_values = np.concatenate([true_train_values, true_val_values])
pred_values = np.concatenate([meta_train_values, meta_val_values])
all_is_train = np.array([True] * len(true_train_values) + [False] * len(true_val_values))

# Sample points for visualization
n_samples = int(0.25 * len(true_values))  # 25% of the dataset
train_ratio = len(true_train_values) / len(true_values)
n_train = int(train_ratio * n_samples)
n_val = n_samples - n_train

# Get indices for sampling from full predictions
train_indices = np.where(all_is_train)[0]
valid_indices = np.where(~all_is_train)[0]
rng = np.random.RandomState(config.rngch_randstate)  # random number generator with fixed seed
train_sample_idx = rng.choice(train_indices, size=n_train, replace=False)
valid_sample_idx = rng.choice(valid_indices, size=n_val, replace=True)

# Combine sampled indices and create mask for plotting
plot_indices = np.concatenate([train_sample_idx, valid_sample_idx])
plot_is_train = np.array([True] * n_train + [False] * n_val)

# Get sampled values for plotting
true_Tg_plot = true_values[plot_indices]
pred_Tg_plot = pred_values[plot_indices]

# Create the parity plot using the same plotter as in the training scripts
plotter = config.ParityPlotter()
plotter.plot(true_Tg_plot, pred_Tg_plot, plot_is_train,
             xlabel=r'Experimental T$_g$ (°C)',
             ylabel=r'Predicted T$_g$ (°C)',
             unit='(°C)',
             title=f'Parity Plot for Glass Transition Temperature (T$_g$) with Meta-Model',
             filename=f'Parity_TgT_{model_name}.png',
             train_metrics=meta_train_metrics,
             val_metrics=meta_val_metrics,
             log_scale=False)


# Define a function to predict Tg for new compounds
def predict_Tg_meta(smiles):
    # Transform SMILES to features
    X_full = feature_gen.transform(smiles)

    # For mode 3, also extract fingerprints and descriptors
    if mode == 3:
        X_fp = X_full[:n_bits]
        X_desc = X_full[n_bits:n_bits + n_descriptors]

    # Get normalized predictions from base models
    with torch.no_grad():
        X_tensor = torch.from_numpy(np.array([X_full])).float()
        ann_pred_norm = ann_model(X_tensor).item()

    cat_pred_norm = catboost_model.predict(np.array([X_full]))[0]

    # Meta-model prediction based on mode
    with torch.no_grad():
        meta_X_preds = torch.tensor([[ann_pred_norm, cat_pred_norm]], dtype=torch.float32)

        if mode == 1:
            meta_pred_norm = meta_model(meta_X_preds).item()
        elif mode == 2:
            meta_X_features = torch.tensor([X_full], dtype=torch.float32)
            meta_pred_norm = meta_model(meta_X_features, meta_X_preds).item()
        elif mode == 3:
            meta_X_fp = torch.tensor([X_fp], dtype=torch.float32)
            meta_X_desc = torch.tensor([X_desc], dtype=torch.float32)
            meta_pred_norm = meta_model(meta_X_fp, meta_X_desc, meta_X_preds).item()

    # Reverse normalization to get actual Tg values
    ann_tg = config.reverse(ann_pred_norm, meta_min_value, meta_max_value)
    cat_tg = config.reverse(cat_pred_norm, meta_min_value, meta_max_value)
    meta_tg = config.reverse(meta_pred_norm, meta_min_value, meta_max_value)

    return ann_tg, cat_tg, meta_tg


# Test predictions on sample compounds
print("\nTesting predictions on sample compounds:")
print("----------------------------------------")
compounds = {
    "OA": {"MonSmiles": "CCCCCCC(C)OC(=O)C=C", "PolSmiles": "*CC(*)C(=O)OC(C)CCCCCC", "val": "-44.0"},
    "OMA": {"MonSmiles": "CCCCCCC(C)OC(=O)C(C)=C", "PolSmiles": "*CC(*)(C)C(=O)OC(C)CCCCCC", "val": "-5.0"},
    "TGA": {"MonSmiles": "CC(CCCC(C)C)CCOC(C=C)=O", "PolSmiles": "*CC(*)C(=O)OCC(CCCC(C)C)C", "val": "-46.0"},
    "TGM": {"MonSmiles": "CC(CCCC(C)C)CCOC(C(C)=C)=O", "PolSmiles": "*CC(*)(C)C(=O)OCC(CCCC(C)C)C", "val": "-13.0"},
    "TDM": {"MonSmiles": "CC(C(OCCCCCCCCCCCCCC)=O)=C", "PolSmiles": "*CC(*)(C)C(=O)OCCCCCCCCCCCCCC", "val": "-22.0"},
    "THFM": {"MonSmiles": "CC(C(OCC1OCCC1)=O)=C", "PolSmiles": "*CC(*)(C)C(=O)OCC1CCCO1", "val": "68.5"},
    "IBOA": {"MonSmiles": "C=CC(OC1CC2CCC1(C)C2(C)C)=O", "PolSmiles": "*CC(*)C(=O)OC1CC2CCC1(C)C2(C)C", "val": "94.0"},
    "IBOMA": {"MonSmiles": "CC(C(OC1CC2CCC1(C)C2(C)C)=O)=C", "PolSmiles": "*CC(*)(C)C(=O)OC1CC2CCC1(C)C2(C)C", "val": "150.0"},
    "MBL": {"MonSmiles": "C=C(CCO1)C1=O", "PolSmiles": "*CC1CC(*)C(=O)O1", "val": "195.0"},
    "BA": {"MonSmiles": "CCCCOC(=O)C=C", "PolSmiles": "*CC(*)C(=O)OCCCC", "val": "-49.0"},
    "MMA": {"MonSmiles": "COC(=O)C(C)=C", "PolSmiles": "*CC(*)(C)C(=O)OC", "val": "108.0"},
    "St": {"MonSmiles": "C=Cc1ccccc1", "PolSmiles": "*CC(*)c1ccccc1", "val": "100.0"}}

# Calculate predictions
table_data = []
meta_errors = []

for name, detail in compounds.items():
    ann_tg, cat_tg, meta_tg = predict_Tg_meta(detail["PolSmiles"])
    exp_val = float(detail["val"])

    # Calculate errors
    ann_error = abs(ann_tg - exp_val)
    cat_error = abs(cat_tg - exp_val)
    meta_error = abs(meta_tg - exp_val)
    meta_errors.append(meta_error)

    table_data.append([name, f"{exp_val:.1f}", f"{ann_tg:.1f} ({ann_error:.1f})", f"{cat_tg:.1f} ({cat_error:.1f})", f"{meta_tg:.1f} ({meta_error:.1f})"])

# Print the table
headers = ["Name", "Tg Exp", "Tg ANN (Err)", "Tg Cat (Err)", f"Tg {model_name} (Err)"]
print(tabulate(table_data, headers=headers, tablefmt="github"))

# Calculate average error
avg_meta_error = sum(meta_errors) / len(meta_errors)
print(f"\nAverage {model_name} Meta-Model Error: {avg_meta_error:.2f}°C")

print(f"\n{model_name} meta-model training completed successfully!")
