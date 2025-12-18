# Training ANN to predict rr
# Version: 2025.10.22

import config
import numpy as np
import pandas as pd
import torch

# Training mode
mode = 'Legacy'  # 'Legacy', 'Encoder' (default), 'Transformer', or 'GANs'

# Importing dataset
df = pd.read_excel("./Data/Dataset_Rr.xlsx")
df = df.loc[(df['r1'] <= 10) & (df['r1'] >= 0.005)]
df = df.loc[(df['r2'] <= 10) & (df['r2'] >= 0.005)]
df = df.reset_index(drop=True)

# Initialize feature generator
feature_gen = config.RrFG(n_bits=config.Full_nbits)

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
FPs = pd.concat([FP1_df, FP2_df], axis=1)

# Saving the parameters
feature_gen.save(f'Features_Rr1_{mode}.pkl')
feature_params = {'n_bits': feature_gen.n_bits,
                  'max_frequency': feature_gen.max_frequency}

# Normalize Rr and combine them
df.r1 = np.log(df.r1)
df.r2 = np.log(df.r2)
Rrs = np.column_stack((df.r1, df.r2))
r_norm, min_value, max_value = config.normalize(Rrs)

# Prepare data
X = torch.from_numpy(np.array(FPs)).float()
y = torch.from_numpy(r_norm).float()

# Training based on selected mode
if mode == 'Legacy' or mode == 'Encoder' or mode == 'Transformer':
    # Use the ModelTrainer with the specified mode
    model_constructor = lambda input_dim: config.ANN_Rr(input_dim, mode=mode)

    trainer = config.ModelTrainer(X=X, y=y, model_class=model_constructor, k_folds=10, batch_size=16, epochs=150, learning_rate=0.00092,
                                  patience=25, min_delta=1e-5, filename=f'Trained_Rr1_{mode}.pt', verbose=True)

elif mode == 'GANs':
    # Use the GANTrainer
    trainer = config.GANTrainer(X=X, y=y, gan_class=config.RrGAN, k_folds=10, batch_size=10, epochs=300, g_lr=0.0001, d_lr=0.0002,
                                patience=100, min_delta=1e-5, filename=f'Trained_Rr1_{mode}.pt', verbose=True)

# Train and Save the model
trainer.train()
trainer.save_model(min_value=min_value, max_value=max_value, ext_data={'feature_params': feature_params, 'mode': mode})
best_train_idx, best_val_idx = trainer.get_best_indices()

# Recreate the model for evaluation
if mode == 'Legacy' or mode == 'Encoder' or mode == 'Transformer':
    model = config.ANN_Rr(input_dim=X.shape[1], mode=mode)
    model.load_state_dict(trainer.best_model_state)
    model.eval()

    # Get predictions
    with torch.no_grad():
        all_train_preds = model(X[best_train_idx])
        all_val_preds = model(X[best_val_idx])

        true_values = np.concatenate([y[best_train_idx].numpy(), y[best_val_idx].numpy()])
        pred_values = np.concatenate([all_train_preds.numpy(), all_val_preds.numpy()])
        all_is_train = np.array([True] * len(best_train_idx) + [False] * len(best_val_idx))

elif mode == 'GANs':
    gan = config.RrGAN(input_dim=X.shape[1])
    generator, _ = gan.get_models()
    generator.load_state_dict(trainer.best_generator_state)
    generator.eval()

    # Generate predictions with fixed noise for reproducibility
    with torch.no_grad():
        fixed_noise_train = torch.randn(len(best_train_idx), gan.latent_dim)
        fixed_noise_val = torch.randn(len(best_val_idx), gan.latent_dim)

        all_train_preds = generator(X[best_train_idx], fixed_noise_train)
        all_val_preds = generator(X[best_val_idx], fixed_noise_val)

        true_values = np.concatenate([y[best_train_idx].numpy(), y[best_val_idx].numpy()])
        pred_values = np.concatenate([all_train_preds.numpy(), all_val_preds.numpy()])
        all_is_train = np.array([True] * len(best_train_idx) + [False] * len(best_val_idx))

# Convert normalized values back to original scale for all modes
true_r1 = np.exp(config.reverse(true_values[:, 0], min_value[0], max_value[0]))
pred_r1 = np.exp(config.reverse(pred_values[:, 0], min_value[0], max_value[0]))
true_r2 = np.exp(config.reverse(true_values[:, 1], min_value[1], max_value[1]))
pred_r2 = np.exp(config.reverse(pred_values[:, 1], min_value[1], max_value[1]))

# Calculate metrics for r1
train_metrics_r1 = config.calc_metrics(true_r1[all_is_train], pred_r1[all_is_train])
valid_metrics_r1 = config.calc_metrics(true_r1[~all_is_train], pred_r1[~all_is_train])

# Calculate metrics for r2
train_metrics_r2 = config.calc_metrics(true_r2[all_is_train], pred_r2[all_is_train])
valid_metrics_r2 = config.calc_metrics(true_r2[~all_is_train], pred_r2[~all_is_train])

# Sample points for visualization
n_samples = int(0.5 * len(true_values))  # 50% of the dataset
train_ratio = (10 - 1) / 10  # same ratio for plotting (k_folds=10)
n_train = int(train_ratio * n_samples)
n_val = n_samples - n_train

# Get indices for sampling from full predictions
train_indices = np.where(all_is_train)[0]
valid_indices = np.where(~all_is_train)[0]
rng = np.random.RandomState(config.rngch_randstate)  # random number generator with fixed seed
train_sample_idx = rng.choice(train_indices, size=n_train, replace=False)
valid_sample_idx = rng.choice(valid_indices, size=n_val, replace=True)

# Combine sampled indices and create a mask for plotting
plot_indices = np.concatenate([train_sample_idx, valid_sample_idx])
plot_is_train = np.array([True] * n_train + [False] * n_val)

# Get sampled values for plotting r1
true_r1_plot = true_r1[plot_indices]
pred_r1_plot = pred_r1[plot_indices]

# Display detailed metrics
config.disp_metrics(train_metrics_r1, valid_metrics_r1, f"{mode.capitalize()} model r1")

# Plotting
plotter = config.ParityPlotter()
plotter.plot(true_r1_plot, pred_r1_plot, plot_is_train,
             xlabel=r'Experimental r$_1$',
             ylabel=r'Predicted r$_1$',
             unit='',
             title=f'Parity Plot for 1st Monomer Reactivity Ratio (r$_1$) with {mode.capitalize()}',
             filename=f'Parity_Rr1_{mode}.png',
             train_metrics=train_metrics_r1, val_metrics=valid_metrics_r1,
             log_scale=True)

