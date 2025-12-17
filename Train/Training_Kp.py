# Training ANN to predict Kp directly
# Version: 2025.10.22

import config
import numpy as np
import pandas as pd
import torch

# Importing dataset
df = pd.read_excel("./Data/Dataset_Kp.xlsx")

# Get and add Fingerprints + Descriptors to dataset
feature_gen = config.KpFG(n_bits=config.Less_nbits)
features = feature_gen.fit_transform(df['S'].tolist())
df['FP'] = features

# Saving the parameters
feature_gen.save('Features_Kp1.pkl')
feature_params = {'n_bits': feature_gen.n_bits,
                  'max_frequency': feature_gen.max_frequency,
                  'descriptor_mins': feature_gen.descriptor_mins,
                  'descriptor_maxs': feature_gen.descriptor_maxs}

# Normalizing Kp
df.Kp = np.log(df.Kp)
normalized = config.normalize(df['Kp'].values)
Kp_norm, min_value, max_value = normalized

# ANN Estimator
X = torch.from_numpy(np.array(features)).float()
y = torch.from_numpy(Kp_norm).float().reshape(-1, 1)

# Use the ModelTrainer for Kp1
trainer = config.ModelTrainer(X=X, y=y, model_class=config.ANN_Kp, k_folds=10,
                              batch_size=2, epochs=150, learning_rate=0.0014,
                              patience=150, min_delta=1e-5, filename='Trained_Kp1.pt', verbose=True)

# Train and Save the model
trainer.train()
trainer.save_model(min_value=min_value, max_value=max_value, ext_data={'feature_params': feature_params})
best_train_idx, best_val_idx = trainer.get_best_indices()

# Recreate the model for evaluation
model = trainer.model_class(input_dim=X.shape[1])
model.load_state_dict(trainer.best_model_state)
model.eval()

# First get predictions for ALL points
with torch.no_grad():
    all_train_preds = model(X[best_train_idx])
    all_val_preds = model(X[best_val_idx])

    true_values = np.concatenate([y[best_train_idx].numpy(), y[best_val_idx].numpy()])
    pred_values = np.concatenate([all_train_preds.numpy(), all_val_preds.numpy()])
    all_is_train = np.array([True] * len(best_train_idx) + [False] * len(best_val_idx))

# Calculate metrics using ALL points
true_Kp = np.exp(config.reverse(true_values, min_value, max_value))
pred_Kp = np.exp(config.reverse(pred_values, min_value, max_value))
train_metrics = config.calc_metrics(true_Kp[all_is_train], pred_Kp[all_is_train])
valid_metrics = config.calc_metrics(true_Kp[~all_is_train], pred_Kp[~all_is_train])

# Sample points for visualization
n_samples = len(true_values)  # All the dataset, since it's small
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
true_Kp_plot = true_Kp[plot_indices]
pred_Kp_plot = pred_Kp[plot_indices]

# Display detailed metrics
config.disp_metrics(train_metrics, valid_metrics, "ANN model Kp")

# Create the parity plot
plotter = config.ParityPlotter()
plotter.plot(true_Kp_plot, pred_Kp_plot, plot_is_train,
             xlabel=r'Experimental K$_p$ (L mol⁻¹ s⁻¹)',
             ylabel=r'Predicted K$_p$ (L mol⁻¹ s⁻¹)',
             unit='(L mol⁻¹ s⁻¹)',
             title=r'Parity Plot for Propagation Rate Constant (K$_p$)',
             filename='Parity_Kp1.png',
             train_metrics=train_metrics, val_metrics=valid_metrics,
             log_scale=True)
