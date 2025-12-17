# Training Gradient Boosting models to predict Ws
# Version: 2025.10.22

import config
import numpy as np
import pandas as pd

# Select model type:
model_type = 'ctb'  # 'ctb' for CatBoost or 'xgb' for XGBoost

# Importing dataset
df = pd.read_excel("./Data/Dataset_Ws.xlsx")
df = df.loc[df['Ws'] >= 1e-10]
df = df.loc[df['S'].str.len() <= 180]
df = df.reset_index(drop=True)
df.Ws = np.log(df.Ws)

# Get and add Fingerprints + Descriptors to dataset
feature_gen = config.WsFG(n_bits=config.Less_nbits)
features = feature_gen.fit_transform(df['S'].tolist())
df['FP'] = features

# Saving the parameters
feature_gen.save('Features_Ws2.pkl')
feature_params = {'n_bits': feature_gen.n_bits,
                  'max_frequency': feature_gen.max_frequency,
                  'descriptor_mins': feature_gen.descriptor_mins,
                  'descriptor_maxs': feature_gen.descriptor_maxs}

# Normalizing Ws
normalized = config.normalize(df['Ws'].values)
Ws_norm, min_value, max_value = normalized

# Convert data to numpy arrays
X = np.array(features)
y = Ws_norm

# Initialize the appropriate model function based on model_type
if model_type == 'ctb':
    model_func = config.CATB_Ws
    model_name = "CatBoost"
    filename = 'Trained_Ws2_ctb.joblib'
elif model_type == 'xgb':
    model_func = config.XGB_Ws
    model_name = "XGBoost"
    filename = 'Trained_Ws2_xgb.joblib'
else:
    raise ValueError(f"Invalid model type: '{model_type}'. Please choose either 'ctb' for CatBoost or 'xgb' for XGBoost.")

# Use the GBMTrainer for Ws
trainer = config.GBMTrainer(X=X, y=y, model_func=model_func, k_folds=10, filename=filename, verbose=True)

# Train and Save the model
trainer.train()
trainer.save_model(min_value=min_value, max_value=max_value, ext_data={'feature_params': feature_params})
best_train_idx, best_val_idx = trainer.get_best_indices()

# First get predictions for ALL points
all_train_preds = trainer.best_model.predict(X[best_train_idx])
all_val_preds = trainer.best_model.predict(X[best_val_idx])

true_values = np.concatenate([y[best_train_idx], y[best_val_idx]])
pred_values = np.concatenate([all_train_preds, all_val_preds])
all_is_train = np.array([True] * len(best_train_idx) + [False] * len(best_val_idx))

# Calculate metrics using ALL points
true_Ws = np.exp(config.reverse(true_values, min_value, max_value))
pred_Ws = np.exp(config.reverse(pred_values, min_value, max_value))
train_metrics = config.calc_metrics(true_Ws[all_is_train], pred_Ws[all_is_train])
valid_metrics = config.calc_metrics(true_Ws[~all_is_train], pred_Ws[~all_is_train])

# Sample points for visualization
n_samples = int(0.15 * len(true_values))  # 15% of the dataset
train_ratio = (trainer.k_folds - 1) / trainer.k_folds  # same ratio for plotting
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
true_Ws_plot = true_Ws[plot_indices]
pred_Ws_plot = pred_Ws[plot_indices]

# Display detailed metrics
config.disp_metrics(train_metrics, valid_metrics, f"{model_name} model Ws")

# Create the parity plot
plotter = config.ParityPlotter()
plotter.plot(true_Ws_plot, pred_Ws_plot, plot_is_train,
             xlabel=r'Experimental W$_s$ (mol L⁻¹)',
             ylabel=r'Predicted W$_s$ (mol L⁻¹)',
             unit='(mol L⁻¹)',
             title=r'Parity Plot for Water Solubility (W$_s$) with ' + model_name,
             filename=f'Parity_Ws2_{model_type}.png',
             train_metrics=train_metrics, val_metrics=valid_metrics,
             log_scale=True)
