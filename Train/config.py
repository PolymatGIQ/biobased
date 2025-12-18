# Configuration file
# Version: 2025.04.01

import os
import time
import copy
import pickle
import torch
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator


### 1. Constants
KFold_randstate = 21
rngch_randstate = 42
Less_nbits = 2020  # For Features with Descriptors [used in Kp, Tg, Ws]
Full_nbits = 2048  # For Features only Fingerprints [used in Rr]


### 2. Features
# 2.0. Base Feature Generator: Descriptors + Fingerprint with Frequency (MFF)
class FeatureGenerator:
    def __init__(self, n_bits):  # Default to 1024 unless specified otherwise
        self.n_bits = n_bits
        self.max_frequency = None
        self.descriptor_mins = None
        self.descriptor_maxs = None
        self.descriptor_ranges = None
        self.constant_features = None

        # Child classes will override this
        self.descriptors = []
        self.descriptor_names = []

    def _generate_raw_fingerprint(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        morgan_gen = GetMorganGenerator(radius=3, fpSize=self.n_bits)
        counts = morgan_gen.GetCountFingerprintAsNumPy(mol)

        return counts

    def _calculate_descriptors(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        return np.array([desc(mol) for desc in self.descriptors])

    def fit(self, smiles_list):
        max_freq = 0
        all_descriptors = []

        # Use the more efficient count-based fingerprint
        for smiles in smiles_list:
            fingerprint = self._generate_raw_fingerprint(smiles)
            max_freq = max(max_freq, np.max(fingerprint))
            descriptors = self._calculate_descriptors(smiles)
            all_descriptors.append(descriptors)

        self.max_frequency = max_freq
        all_descriptors = np.array(all_descriptors)

        # Calculate statistics
        self.descriptor_mins = np.nanmin(all_descriptors, axis=0)
        self.descriptor_maxs = np.nanmax(all_descriptors, axis=0)
        self.descriptor_ranges = self.descriptor_maxs - self.descriptor_mins

        # Identify constant features (where max equals min)
        self.constant_features = np.isclose(self.descriptor_ranges, 0, atol=1e-10)

        # Replace zero ranges with 1 to avoid division by zero
        self.descriptor_ranges[self.constant_features] = 1.0

        return self

    def transform(self, smiles):
        if self.max_frequency is None or self.descriptor_mins is None:
            raise ValueError("FeatureGenerator needs to be fit before transform can be called")

        fingerprint = self._generate_raw_fingerprint(smiles)
        normalized_fingerprint = fingerprint / (self.max_frequency if self.max_frequency > 0 else 1.0)
        descriptors = self._calculate_descriptors(smiles)

        # Handle NaN values in descriptors (Normalize safely, Set constant features to 0.5, etc.)
        descriptors = np.nan_to_num(descriptors, nan=self.descriptor_mins)
        normalized_descriptors = (descriptors - self.descriptor_mins) / self.descriptor_ranges
        normalized_descriptors[self.constant_features] = 0.5
        normalized_descriptors = np.clip(normalized_descriptors, 0, 1)

        return np.concatenate([normalized_fingerprint, normalized_descriptors])

    def fit_transform(self, smiles_list):
        self.fit(smiles_list)
        return [self.transform(smiles) for smiles in smiles_list]

    def save(self, filename):
        os.makedirs("./Files", exist_ok=True)
        with open(f"./Files/{filename}", 'wb') as f:
            pickle.dump({
                'max_frequency': self.max_frequency,
                'descriptor_mins': self.descriptor_mins,
                'descriptor_maxs': self.descriptor_maxs,
                'n_bits': self.n_bits}, f)

    @classmethod
    def load(cls, source):
        if isinstance(source, str):  # Legacy behavior - load from pkl file
            with open(source, 'rb') as f:
                data = pickle.load(f)
        elif isinstance(source, dict):  # New behavior - load from model file
            data = source
        else:
            raise TypeError("Source must be either a filepath string or a parameter dictionary")

        feature_gen = cls(n_bits=data.get('n_bits', Less_nbits))
        feature_gen.max_frequency = data['max_frequency']
        feature_gen.descriptor_mins = data.get('descriptor_mins', None)
        feature_gen.descriptor_maxs = data.get('descriptor_maxs', None)

        # Only calculate descriptor ranges if descriptors exist
        if feature_gen.descriptor_mins is not None and feature_gen.descriptor_maxs is not None:
            feature_gen.descriptor_ranges = feature_gen.descriptor_maxs - feature_gen.descriptor_mins
            feature_gen.constant_features = np.isclose(feature_gen.descriptor_ranges, 0, atol=1e-10)
            feature_gen.descriptor_ranges[feature_gen.constant_features] = 1.0
        else:  # For feature generators without descriptors (like RrFG)
            feature_gen.descriptor_ranges = None
            feature_gen.constant_features = None

        return feature_gen

    def get_feature_size(self):
        return self.n_bits + len(self.descriptors)

    def get_feature_names(self):
        return [f"bit_{i}" for i in range(self.n_bits)] + self.descriptor_names


# 2.1. Feature generator specialized for Rr prediction
class RrFG(FeatureGenerator):
    def __init__(self, n_bits):
        super().__init__(n_bits)
        # Empty descriptors list
        self.descriptors = []
        self.descriptor_names = []

    def transform(self, smiles):
        if self.max_frequency is None:
            raise ValueError("FeatureGenerator needs to be fit before transform can be called")

        fingerprint = self._generate_raw_fingerprint(smiles)
        normalized_fingerprint = fingerprint / (self.max_frequency if self.max_frequency > 0 else 1.0)

        # Just return the fingerprint (no descriptors)
        return normalized_fingerprint

    def fit_transform(self, smiles_list):
        self.fit(smiles_list)
        return [self.transform(smiles) for smiles in smiles_list]


# 2.2. Feature generator specialized for Kp prediction
class KpFG(FeatureGenerator):
    def __init__(self, n_bits):
        super().__init__(n_bits)
        self.descriptors = [
            Descriptors.fr_NH1,
            Descriptors.fr_alkyl_carbamate,
            Descriptors.Phi,
            Descriptors.qed,
            Descriptors.fr_unbrch_alkane,
            Descriptors.MinAbsPartialCharge,
            Descriptors.PEOE_VSA6,
            Descriptors.PEOE_VSA3,
            Descriptors.Kappa3,
            Descriptors.NumHDonors,
            Descriptors.EState_VSA5,
            Descriptors.SlogP_VSA10,
            Descriptors.VSA_EState7,
            Descriptors.AUTOCORR2D_15,
            Descriptors.MaxAbsPartialCharge,
            Descriptors.Kappa1,
            Descriptors.NHOHCount,
            Descriptors.MolLogP,
            Descriptors.SMR_VSA6,
            Descriptors.BCUT2D_MRLOW,
            Descriptors.Chi1n,
            Descriptors.SlogP_VSA5,
            Descriptors.PEOE_VSA14,
            Descriptors.MaxAbsEStateIndex,
            Descriptors.FractionCSP3,
            Descriptors.PEOE_VSA2,
            Descriptors.SlogP_VSA1,
            Descriptors.fr_allylic_oxid
        ]

        self.descriptor_names = [desc.__name__ for desc in self.descriptors]


# 2.3. Feature generator specialized for Tg prediction
class TgFG(FeatureGenerator):
    def __init__(self, n_bits):
        super().__init__(n_bits)
        self.descriptors = [
            Descriptors.RingCount,
            Descriptors.FractionCSP3,
            Descriptors.AUTOCORR2D_26,
            Descriptors.PEOE_VSA14,
            Descriptors.AUTOCORR2D_10,
            Descriptors.SlogP_VSA1,
            Descriptors.BalabanJ,
            Descriptors.VSA_EState6,
            Descriptors.NumHeteroatoms,
            Descriptors.fr_bicyclic,
            Descriptors.Chi3n,
            Descriptors.SMR_VSA10,
            Descriptors.AvgIpc,
            Descriptors.VSA_EState2,
            Descriptors.fr_amide,
            Descriptors.fr_imide,
            Descriptors.fr_aniline,
            Descriptors.NumAliphaticRings,
            Descriptors.SlogP_VSA7,
            Descriptors.fr_NH0,
            Descriptors.SMR_VSA9,
            Descriptors.MaxEStateIndex,
            Descriptors.EState_VSA6,
            Descriptors.TPSA,
            Descriptors.SlogP_VSA8,
            Descriptors.EState_VSA2,
            Descriptors.SMR_VSA5,
            Descriptors.Chi2v
        ]
        self.descriptor_names = [desc.__name__ for desc in self.descriptors]


# 2.4. Feature generator specialized for Ws prediction
class WsFG(FeatureGenerator):
    def __init__(self, n_bits):
        super().__init__(n_bits)
        self.descriptors = [
            Descriptors.AUTOCORR2D_27,
            Descriptors.AUTOCORR2D_13,
            Descriptors.AvgIpc,
            Descriptors.MolMR,
            Descriptors.MolLogP,
            Descriptors.Chi2v,
            Descriptors.Chi3n,
            Descriptors.BertzCT,
            Descriptors.RingCount,
            Descriptors.PEOE_VSA6,
            Descriptors.PEOE_VSA7,
            Descriptors.SlogP_VSA5,
            Descriptors.Kappa2,
            Descriptors.qed,
            Descriptors.SMR_VSA10,
            Descriptors.SMR_VSA7,
            Descriptors.EState_VSA4,
            Descriptors.NumAromaticCarbocycles,
            Descriptors.NumAromaticRings,
            Descriptors.EState_VSA3,
            Descriptors.SlogP_VSA4,
            Descriptors.EState_VSA5,
            Descriptors.VSA_EState7,
            Descriptors.HallKierAlpha,
            Descriptors.NumAliphaticCarbocycles,
            Descriptors.MaxAbsEStateIndex,
            Descriptors.NumAliphaticRings,
            Descriptors.EState_VSA8
            ]

        self.descriptor_names = [desc.__name__ for desc in self.descriptors]


### 3. Machine Learning
# 3.1. ANN for Direct Kp
class ANN_Kp(torch.nn.Module):
    def __init__(self, input_dim):
        super(ANN_Kp, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(64, 96),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.0),
            torch.nn.Linear(96, 48),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.0),
            torch.nn.Linear(48, 1))

    def forward(self, x):
        return self.model(x).view(-1, 1)


# 3.2 ANN for Arrhenius Kp
class ANN_Ar(torch.nn.Module):
    def __init__(self, input_dim):
        super(ANN_Ar, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 112),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.0),
            torch.nn.Linear(112, 80),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.0),
            torch.nn.Linear(80, 2))

    def forward(self, x):
        return self.model(x)


# 3.3. ANN for Rr with multiple modes
class ANN_Rr(torch.nn.Module):
    def __init__(self, input_dim, mode='Encoder'):
        super(ANN_Rr, self).__init__()
        self.mode = mode
        self.nbits = input_dim // 2

        if mode == 'Legacy':
            # Vanilla approach - direct prediction
            self.model = torch.nn.Sequential(
                torch.nn.Linear(input_dim, 80),
                torch.nn.ReLU(),
                torch.nn.Linear(80, 40),
                torch.nn.ReLU(),
                torch.nn.Linear(40, 2))

        elif mode == 'Encoder' or mode is None:  # Default is Encoder mode
            # Encoding approach
            condensed_bits = 24
            self.Qe = torch.nn.Sequential(
                torch.nn.Linear(self.nbits, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, condensed_bits),
                torch.nn.ReLU())

            self.Rr = torch.nn.Sequential(
                torch.nn.Linear(2 * condensed_bits, 240),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.2),
                torch.nn.Linear(240, 128),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.0),
                torch.nn.Linear(128, 32),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.0),
                torch.nn.Linear(32, 2))

        elif mode == 'Transformer':
            # Transformer approach
            transformer_hidden = 128

            # Embedding layers for each monomer
            self.embedding1 = torch.nn.Linear(self.nbits, transformer_hidden)
            self.embedding2 = torch.nn.Linear(self.nbits, transformer_hidden)

            # Transformer encoder layers
            encoder_layer = torch.nn.TransformerEncoderLayer(d_model=transformer_hidden, nhead=4,
                                                             dim_feedforward=transformer_hidden * 2,
                                                             dropout=0.1, batch_first=True)

            self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=3)

            # Prediction head
            self.output = torch.nn.Sequential(
                torch.nn.Linear(transformer_hidden * 2, transformer_hidden),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.1),
                torch.nn.Linear(transformer_hidden, 2))

    def forward(self, x):
        if self.mode == 'Legacy':
            return self.model(x)

        elif self.mode == 'Encoder' or self.mode is None:  # Default to Encoder mode
            M1QE = self.Qe(x[:, :self.nbits])
            M2QE = self.Qe(x[:, self.nbits:])
            xx = torch.cat([M1QE, M2QE], 1)
            return self.Rr(xx)

        elif self.mode == 'Transformer':
            # Split input into two monomers and embed each
            x1 = self.embedding1(x[:, :self.nbits]).unsqueeze(1)
            x2 = self.embedding2(x[:, self.nbits:]).unsqueeze(1)

            # Process through transformer
            x1 = self.transformer(x1).squeeze(1)
            x2 = self.transformer(x2).squeeze(1)

            # Concatenate and predict
            combined = torch.cat([x1, x2], dim=1)
            return self.output(combined)


# 3.4. GAN for Rr prediction
class Generator(torch.nn.Module):
    def __init__(self, latent_dim, fingerprint_dim):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.fingerprint_dim = fingerprint_dim

        self.model = torch.nn.Sequential(
            torch.nn.Linear(latent_dim + fingerprint_dim * 2, 512),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Linear(512, 256),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Linear(256, 128),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Linear(128, 64),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Linear(64, 2),
            torch.nn.Sigmoid())  # Output in [0,1] range

    def forward(self, fingerprints, noise=None):
        if noise is None:
            noise = torch.randn(fingerprints.size(0), self.latent_dim, device=fingerprints.device)
        x = torch.cat([fingerprints, noise], dim=1)
        return self.model(x)


class Discriminator(torch.nn.Module):
    def __init__(self, fingerprint_dim):
        super(Discriminator, self).__init__()

        self.model = torch.nn.Sequential(
            torch.nn.Linear(fingerprint_dim * 2 + 2, 256),  # Fingerprints + Rr values
            torch.nn.LeakyReLU(0.2),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, 128),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(128, 64),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(64, 1),
            torch.nn.Sigmoid())  # Real/fake probability

    def forward(self, fingerprints, rr_values):
        x = torch.cat([fingerprints, rr_values], dim=1)
        return self.model(x)


class RrGAN:
    def __init__(self, input_dim, latent_dim=100):
        self.generator = Generator(latent_dim, input_dim // 2)
        self.discriminator = Discriminator(input_dim // 2)
        self.latent_dim = latent_dim
        self.input_dim = input_dim

    def get_models(self):
        return self.generator, self.discriminator


# 3.5. CatBoost for Rr
def CATB_Rr():
    from catboost import CatBoostRegressor
    return CatBoostRegressor(
        iterations=4000,
        learning_rate=0.03,
        depth=10,
        l2_leaf_reg=4,
        loss_function='MultiRMSE',
        random_seed=20,
        verbose=500,
        allow_writing_files=False)


# 3.6. XGBoost for Rr
def XGB_Rr():
    from xgboost import XGBRegressor
    return XGBRegressor(
        n_estimators=1200,
        learning_rate=0.03,
        max_depth=8,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.2,
        reg_alpha=3,
        reg_lambda=1,
        objective='reg:squarederror',
        random_state=20,
        verbosity=1)


# 3.7. ANN for Tg
class ANN_Tg(torch.nn.Module):
    def __init__(self, input_dim):
        super(ANN_Tg, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 496),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.4),
            torch.nn.Linear(496, 176),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.0),
            torch.nn.Linear(176, 96),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(96, 1))

    def forward(self, x):
        return self.model(x).view(-1, 1)


# 3.8. CatBoost for Tg
def CATB_Tg():
    from catboost import CatBoostRegressor
    return CatBoostRegressor(
        iterations=2000,
        learning_rate=0.04,
        depth=10,
        l2_leaf_reg=5,
        loss_function='RMSE',
        random_seed=20,
        verbose=200,
        allow_writing_files=False)


# 3.9. XGBoost for Tg
def XGB_Tg():
    from xgboost import XGBRegressor
    return XGBRegressor(
        n_estimators=1200,
        learning_rate=0.03,
        max_depth=9,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.2,
        reg_alpha=3,
        reg_lambda=1,
        objective='reg:squarederror',
        random_state=20,
        verbosity=1)


# 3.10. ANN for Ws
class ANN_Ws(torch.nn.Module):
    def __init__(self, input_dim):
        super(ANN_Ws, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 208),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(208, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.0),
            torch.nn.Linear(128, 48),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(48, 1))

    def forward(self, x):
        return self.model(x).view(-1, 1)


# 3.11. CatBoost for Ws
def CATB_Ws():
    from catboost import CatBoostRegressor
    return CatBoostRegressor(
        iterations=2000,
        learning_rate=0.04,
        depth=10,
        l2_leaf_reg=5,
        loss_function='RMSE',
        random_seed=20,
        verbose=200,
        allow_writing_files=False)


# 3.12. XGBoost for Ws
def XGB_Ws():
    from xgboost import XGBRegressor
    return XGBRegressor(
        n_estimators=1200,
        learning_rate=0.03,
        max_depth=9,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.2,
        reg_alpha=3,
        reg_lambda=1,
        objective='reg:squarederror',
        random_state=20,
        verbosity=1)


# 3.13. ANN for Meta-Models
class MetaModelANN(torch.nn.Module):
    def __init__(self):
        super(MetaModelANN, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(2, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1))

    def forward(self, x):
        return self.model(x)


### 4. Normalization
# 4.1. Normalize the output data
def normalize(data):
    data = np.asarray(data)
    min_value = np.min(data, axis=0)
    max_value = np.max(data, axis=0)
    norm_data = (data - min_value) / (max_value - min_value)
    return norm_data, min_value, max_value


# 4.2. Reverse the normalization process
def reverse(norm_data, min_value, max_value):
    orgn_data = norm_data * (max_value - min_value) + min_value
    return orgn_data


### 5. Model Training
# 5.1. Neural Network Architecture
class ModelTrainer:
    def __init__(self, X, y, model_class, k_folds,
                 batch_size, epochs, learning_rate,
                 patience, min_delta, filename, verbose):

        self.X = X
        self.y = y
        self.model_class = model_class
        self.k_folds = k_folds
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.patience = patience
        self.min_delta = min_delta
        self.filename = filename
        self.verbose = verbose

        # Results storage
        self.best_model_state = None
        self.best_val_loss = float('inf')
        self.best_fold = None
        self.best_train_idx = None
        self.best_val_idx = None
        self.min_value = None
        self.max_value = None

    # Print log messages if verbose mode is enabled
    def log(self, message):
        if self.verbose:
            print(message)

    # Execute the full training process with K-fold cross-validation
    def train(self):
        from sklearn.model_selection import KFold

        # Define K-Fold cross-validator
        kf = KFold(n_splits=self.k_folds, shuffle=True, random_state=KFold_randstate)

        self.log('|-- Training log ------------------------------------')

        for fold, (train_idx, val_idx) in enumerate(kf.split(self.X)):
            self.log(f'\nFold {fold + 1}/{self.k_folds}')

            # Create data for this fold
            X_train, X_val = self.X[train_idx], self.X[val_idx]
            y_train, y_val = self.y[train_idx], self.y[val_idx]

            # Create datasets and dataloaders
            trainloader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(X_train, y_train),
                batch_size=self.batch_size,
                shuffle=True)
            validloader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(X_val, y_val),
                batch_size=self.batch_size,
                shuffle=False)

            # Create model, loss function, and optimizer
            model = self.model_class(input_dim=self.X.shape[1])
            loss_function = torch.nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)

            # Early stopping variables
            best_epoch_val_loss = float('inf')
            best_epoch_model = None
            epochs_without_improvement = 0

            # Training loop
            for epoch in range(self.epochs):
                start_time = time.time()
                model.train()
                total_train_loss = 0.
                n_train_batches = 0

                # Training loop
                for xL, yL in trainloader:
                    optimizer.zero_grad()
                    pred = model(xL)
                    trainloss = loss_function(pred, yL)
                    trainloss.backward()
                    optimizer.step()
                    total_train_loss += trainloss.item()
                    n_train_batches += 1

                # Validation loop
                model.eval()
                total_val_loss = 0.
                n_val_batches = 0

                with torch.no_grad():
                    for xV, yV in validloader:
                        pred = model(xV)
                        val_loss_batch = loss_function(pred, yV)
                        total_val_loss += val_loss_batch.item()
                        n_val_batches += 1

                avg_train_loss = total_train_loss / n_train_batches
                avg_val_loss = total_val_loss / n_val_batches

                end_time = time.time()
                if (epoch + 1) % 20 == 0 or epoch == 0:
                    self.log(f'Epoch {epoch + 1}/{self.epochs}:\n'
                             f'time: {(end_time - start_time):.3f}s'
                             f' ---> Train_Loss: {avg_train_loss:.5f}'
                             f' | Val_Loss: {avg_val_loss:.5f}')

                # Check if this is the best epoch for this fold
                if avg_val_loss < best_epoch_val_loss - self.min_delta:
                    best_epoch_val_loss = avg_val_loss
                    best_epoch_model = copy.deepcopy(model.state_dict())
                    epochs_without_improvement = 0
                    if (epoch + 1) % 20 == 0 or epoch == 0:
                        self.log(f"New best epoch! Validation loss: {best_epoch_val_loss:.5f}")
                else:
                    epochs_without_improvement += 1

                # Early stopping check
                if epochs_without_improvement >= self.patience:
                    self.log(f"\nEarly stopping triggered after {epoch + 1} epochs")
                    break

            # After training this fold, calculate final predictions
            model.load_state_dict(best_epoch_model)
            model.eval()

            # Calculate final validation loss
            final_val_loss = best_epoch_val_loss

            # If this is the best model across all folds, save it
            if final_val_loss < self.best_val_loss:
                self.log(f"\nNew best model found in fold {fold + 1}!")
                self.log(f"Validation loss improved from {self.best_val_loss:.5f} to {final_val_loss:.5f}")
                self.best_val_loss = final_val_loss
                self.best_model_state = copy.deepcopy(best_epoch_model)
                self.best_fold = fold + 1

                # Store indices for this fold
                self.best_train_idx = train_idx
                self.best_val_idx = val_idx

        self.log(f"\nFinished successfully. Best model (from fold {self.best_fold}) saved.")
        return self

    def save_model(self, min_value=None, max_value=None, ext_data=None):
        # Prepare save dictionary
        save_dict = {
            'model_state_dict': self.best_model_state,
            'min_value': min_value,
            'max_value': max_value,
            'input_dim': self.X.shape[1],
            'best_fold': self.best_fold,
            'best_val_loss': self.best_val_loss}

        # Update with any additional data
        if ext_data:
            save_dict.update(ext_data)

        # Save the model
        os.makedirs("./Files", exist_ok=True)
        torch.save(save_dict, f"./Files/{self.filename}")
        self.log("Model has been saved.")

        # Store normalization values
        self.min_value = min_value
        self.max_value = max_value

        return self

    # Return the train and validation indices for the best fold
    def get_best_indices(self):
        return self.best_train_idx, self.best_val_idx


# 5.2. Gradient Boosting Architecture
class GBMTrainer:
    def __init__(self, X, y, model_func, k_folds, filename, verbose):
        self.X = X
        self.y = y
        self.model_func = model_func
        self.k_folds = k_folds
        self.filename = filename
        self.verbose = verbose

        # Results storage
        self.best_model = None
        self.best_val_loss = float('inf')
        self.best_fold = None
        self.best_train_idx = None
        self.best_val_idx = None
        self.min_value = None
        self.max_value = None

    # Print log messages if verbose mode is enabled
    def log(self, message):
        if self.verbose:
            print(message)

    # Execute the full training process with K-fold cross-validation
    def train(self):
        from sklearn.model_selection import KFold
        # Define K-Fold cross-validator
        kf = KFold(n_splits=self.k_folds, shuffle=True, random_state=KFold_randstate)

        self.log('|-- Training log ------------------------------------')

        for fold, (train_idx, val_idx) in enumerate(kf.split(self.X)):
            self.log(f'\nFold {fold + 1}/{self.k_folds}')

            # Create data for this fold
            X_train, X_val = self.X[train_idx], self.X[val_idx]
            y_train, y_val = self.y[train_idx], self.y[val_idx]

            # Initialize and train the model
            start_time = time.time()
            model = self.model_func()

            if 'CatBoost' in model.__class__.__name__:
                model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=100, use_best_model=True)
            elif 'XGB' in model.__class__.__name__:
                model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)], verbose=100)
            else:
                raise TypeError("Unsupported model type")

            # Get predictions
            train_preds = model.predict(X_train)
            val_preds = model.predict(X_val)

            # Calculate validation loss (MSE)
            val_loss = np.mean((y_val - val_preds) ** 2)

            end_time = time.time()
            self.log(f'Training completed in {(end_time - start_time):.3f}s')
            self.log(f'Validation MSE: {val_loss:.5f}')

            # If this is the best model so far, save it
            if val_loss < self.best_val_loss:
                self.log(f"\nNew best model found in fold {fold + 1}!")
                self.log(f"Validation loss improved from {self.best_val_loss:.5f} to {val_loss:.5f}")
                self.best_val_loss = val_loss
                self.best_model = copy.deepcopy(model)
                self.best_fold = fold + 1
                self.best_train_idx = train_idx
                self.best_val_idx = val_idx

        self.log(f"\nFinished successfully. Best model (from fold {self.best_fold}) saved.")
        return self

    def save_model(self, min_value=None, max_value=None, ext_data=None):
        # Prepare save dictionary
        save_dict = {'model': self.best_model, 'min_value': min_value, 'max_value': max_value, 'input_dim': self.X.shape[1], 'best_fold': self.best_fold, 'best_val_loss': self.best_val_loss}

        # Update with any additional data
        if ext_data:
            save_dict.update(ext_data)

        # Save the model
        import joblib
        os.makedirs("./Files", exist_ok=True)
        joblib.dump(save_dict, f"./Files/{self.filename}", compress=('gzip', 9))
        self.log("Model has been saved.")

        # Store normalization values
        self.min_value = min_value
        self.max_value = max_value

        return self

    # Return the train and validation indices for the best fold
    def get_best_indices(self):
        return self.best_train_idx, self.best_val_idx


# 5.3. GANs Trainer
class GANTrainer:
    def __init__(self, X, y, gan_class, k_folds, batch_size, epochs,
                 g_lr, d_lr, patience, min_delta, filename, verbose):

        self.X = X
        self.y = y
        self.gan_class = gan_class
        self.k_folds = k_folds
        self.batch_size = batch_size
        self.epochs = epochs
        self.g_lr = g_lr  # Generator learning rate
        self.d_lr = d_lr  # Discriminator learning rate
        self.patience = patience
        self.min_delta = min_delta
        self.filename = filename
        self.verbose = verbose

        # Results storage
        self.best_generator_state = None
        self.best_discriminator_state = None
        self.best_val_loss = float('inf')
        self.best_fold = None
        self.best_train_idx = None
        self.best_val_idx = None
        self.min_value = None
        self.max_value = None
        self.latent_dim = None

    def log(self, message):
        if self.verbose:
            print(message)

    def train(self):
        from sklearn.model_selection import KFold

        # Define K-Fold cross-validator
        kf = KFold(n_splits=self.k_folds, shuffle=True, random_state=KFold_randstate)

        self.log('|-- GAN Training log ------------------------------------')

        for fold, (train_idx, val_idx) in enumerate(kf.split(self.X)):
            self.log(f'\nFold {fold + 1}/{self.k_folds}')

            # Create data for this fold
            X_train, X_val = self.X[train_idx], self.X[val_idx]
            y_train, y_val = self.y[train_idx], self.y[val_idx]

            # Create datasets and dataloaders
            trainloader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(X_train, y_train),
                batch_size=self.batch_size,
                shuffle=True)
            validloader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(X_val, y_val),
                batch_size=self.batch_size,
                shuffle=False)

            # Initialize GAN
            gan = self.gan_class(input_dim=self.X.shape[1])
            generator, discriminator = gan.get_models()
            self.latent_dim = gan.latent_dim

            # Define loss functions and optimizers
            adversarial_loss = torch.nn.BCELoss()
            mse_loss = torch.nn.MSELoss()

            generator_optimizer = torch.optim.Adam(generator.parameters(), lr=self.g_lr, betas=(0.5, 0.999))
            discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=self.d_lr, betas=(0.5, 0.999))

            # Early stopping variables
            best_epoch_val_loss = float('inf')
            best_epoch_g_model = None
            best_epoch_d_model = None
            epochs_without_improvement = 0

            # Training loop
            for epoch in range(self.epochs):
                start_time = time.time()

                # Training metrics
                train_g_loss = 0.0
                train_d_loss = 0.0
                train_mse_loss = 0.0
                n_batches = 0

                for fingerprints, real_rr in trainloader:
                    batch_size = fingerprints.size(0)
                    n_batches += 1

                    # Ground truths
                    valid = torch.ones(batch_size, 1)
                    fake = torch.zeros(batch_size, 1)

                    ### Train Discriminator
                    discriminator_optimizer.zero_grad()

                    # Generate noise and predictions
                    noise = torch.randn(batch_size, self.latent_dim)
                    gen_rr = generator(fingerprints, noise)

                    # Measure discriminator ability to classify real vs generated samples
                    real_loss = adversarial_loss(discriminator(fingerprints, real_rr), valid)
                    fake_loss = adversarial_loss(discriminator(fingerprints, gen_rr.detach()), fake)
                    d_loss = (real_loss + fake_loss) / 2

                    d_loss.backward()
                    discriminator_optimizer.step()

                    ### Train Generator
                    generator_optimizer.zero_grad()

                    # Generate new predictions
                    noise = torch.randn(batch_size, self.latent_dim)
                    gen_rr = generator(fingerprints, noise)

                    # Adversarial loss (fool discriminator)
                    g_adv_loss = adversarial_loss(discriminator(fingerprints, gen_rr), valid)

                    # Content loss (match real values)
                    g_content_loss = mse_loss(gen_rr, real_rr)

                    # Total generator loss (weighted sum)
                    g_loss = 0.1 * g_adv_loss + 0.9 * g_content_loss

                    g_loss.backward()
                    generator_optimizer.step()

                    # Track losses
                    train_g_loss += g_loss.item()
                    train_d_loss += d_loss.item()
                    train_mse_loss += g_content_loss.item()

                # Calculate average training losses
                avg_g_loss = train_g_loss / n_batches
                avg_d_loss = train_d_loss / n_batches
                avg_mse_loss = train_mse_loss / n_batches

                # Validation
                generator.eval()
                val_mse_loss = 0.0
                n_val_batches = 0

                with torch.no_grad():
                    for val_fp, val_rr in validloader:
                        n_val_batches += 1
                        noise = torch.randn(val_fp.size(0), self.latent_dim)
                        val_gen_rr = generator(val_fp, noise)
                        val_mse_loss += mse_loss(val_gen_rr, val_rr).item()

                avg_val_loss = val_mse_loss / n_val_batches

                end_time = time.time()

                if (epoch + 1) % 20 == 0 or epoch == 0:
                    self.log(f'Epoch {epoch + 1}/{self.epochs}:\n'
                             f'time: {(end_time - start_time):.3f}s'
                             f' ---> G_Loss: {avg_g_loss:.5f}'
                             f' | D_Loss: {avg_d_loss:.5f}'
                             f' | Train_MSE: {avg_mse_loss:.5f}'
                             f' | Val_MSE: {avg_val_loss:.5f}')

                # Check for improvement
                if avg_val_loss < best_epoch_val_loss - self.min_delta:
                    best_epoch_val_loss = avg_val_loss
                    best_epoch_g_model = copy.deepcopy(generator.state_dict())
                    best_epoch_d_model = copy.deepcopy(discriminator.state_dict())
                    epochs_without_improvement = 0
                    if (epoch + 1) % 20 == 0 or epoch == 0:
                        self.log(f"New best epoch! Validation MSE: {best_epoch_val_loss:.5f}")
                else:
                    epochs_without_improvement += 1

                # Early stopping check
                if epochs_without_improvement >= self.patience:
                    self.log(f"\nEarly stopping triggered after {epoch + 1} epochs")
                    break

            # After training this fold, compare with best so far
            if best_epoch_val_loss < self.best_val_loss:
                self.log(f"\nNew best model found in fold {fold + 1}!")
                self.log(f"Validation MSE improved from {self.best_val_loss:.5f} to {best_epoch_val_loss:.5f}")
                self.best_val_loss = best_epoch_val_loss
                self.best_generator_state = copy.deepcopy(best_epoch_g_model)
                self.best_discriminator_state = copy.deepcopy(best_epoch_d_model)
                self.best_fold = fold + 1

                # Store indices for this fold
                self.best_train_idx = train_idx
                self.best_val_idx = val_idx

        self.log(f"\nFinished successfully. Best GAN model (from fold {self.best_fold}) saved.")
        return self

    def save_model(self, min_value=None, max_value=None, ext_data=None):
        # Prepare save dictionary
        save_dict = {
            'generator_state_dict': self.best_generator_state,
            'discriminator_state_dict': self.best_discriminator_state,
            'min_value': min_value,
            'max_value': max_value,
            'input_dim': self.X.shape[1],
            'latent_dim': self.latent_dim,
            'best_fold': self.best_fold,
            'best_val_loss': self.best_val_loss}

        # Update with any additional data
        if ext_data:
            save_dict.update(ext_data)

        # Save the model
        os.makedirs("./Files", exist_ok=True)
        torch.save(save_dict, f"./Files/{self.filename}")
        self.log("GAN model has been saved.")

        # Store normalization values
        self.min_value = min_value
        self.max_value = max_value

        return self

    def get_best_indices(self):
        return self.best_train_idx, self.best_val_idx


### 6. Metrics
# 6.1. Calculating the Metrics
def calc_metrics(y_true, y_pred):
    metrics = {}
    metrics['MAE'] = np.mean(np.abs(y_true - y_pred))  # Mean Absolute Error
    metrics['MSE'] = np.mean((y_true - y_pred) ** 2)  # Mean Squared Error
    metrics['RMSE'] = np.sqrt(metrics['MSE'])  # Root Mean Squared Error
    metrics['R2'] = 1 - ((np.sum((y_true - y_pred) ** 2)) / (np.sum((y_true - np.mean(y_true)) ** 2)))  # R-squared

    # Loged Errors
    with np.errstate(divide='ignore', invalid='ignore'):
        # Only compute log for positive values
        mask_positive = (y_true > 0) & (y_pred > 0)
        if np.any(mask_positive):
            log_squared_errors = (np.log(y_true[mask_positive]) - np.log(y_pred[mask_positive])) ** 2
            metrics['MLSE'] = np.mean(log_squared_errors)  # Mean Log Squared Error
            metrics['RMLSE'] = np.sqrt(metrics['MLSE'])  # Root Mean Log Squared Error
        else:
            metrics['MLSE'] = np.nan
            metrics['RMLSE'] = np.nan

    # Relative Errors
    with np.errstate(divide='ignore', invalid='ignore'):
        # Calculate only where y_true is not zero, replacing inf/nan values with 0
        relative_errors = np.abs((y_true - y_pred) / y_true)
        mask = ~np.isfinite(relative_errors)
        relative_errors[mask] = np.nan
        metrics['MRE'] = np.nanmean(relative_errors)  # Mean Relative Error
        metrics['MAPE'] = metrics['MRE'] * 100  # Mean Absolute Percentage Error

    return metrics


# 6.2. Reporting the Metrics
def disp_metrics(train_metrics, val_metrics, metric_name=""):
    if metric_name:
        print(f"\nMetrics for {metric_name}")

    headers = ["Metric", "Training", "Validation"]
    table_data = []

    for metric in ['MAE', 'MSE', 'RMSE', 'MRE', 'MAPE', 'R2']:
        row = [metric, train_metrics[metric], val_metrics[metric]]
        table_data.append(row)

    # Format the table
    from tabulate import tabulate
    print(tabulate(table_data, headers=headers, floatfmt=".4f", tablefmt="simple"))


### 7. Figures
# 7.1. General setting
lb_fsize = 17
lg_fsize = 13
dpi = 300


# 7.2. ParityPlotter
class ParityPlotter:
    def __init__(self, figsize=(8, 8), dpi=dpi, lb_fsize=lb_fsize, lg_fsize=lg_fsize):
        self.figsize = figsize
        self.dpi = dpi
        self.lb_fsize = lb_fsize  # Label font size
        self.lg_fsize = lg_fsize  # Legend font size

    # Creates a parity plot with marginal KDE distributions.
    def plot(self, true_values, pred_values, is_train, xlabel, ylabel, unit, title, filename,
             train_metrics, val_metrics, log_scale=False, kde_bandwidth=None, axlim=None):

        # Import libraries
        import seaborn as sns
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        # Prepare data
        true_values, pred_values, is_train = map(np.asarray, [true_values, pred_values, is_train])
        true_values, pred_values = true_values.flatten(), pred_values.flatten()

        # Setup figure with gridspec
        fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        gs = gridspec.GridSpec(4, 4, height_ratios=[0.6, 3, 3, 3], width_ratios=[3, 3, 3, 0.6])

        # Create axes for main plot and marginal distributions
        ax_main = fig.add_subplot(gs[1:, :-1])
        ax_x_marginal = fig.add_subplot(gs[0, :-1], sharex=ax_main)
        ax_y_marginal = fig.add_subplot(gs[1:, -1], sharey=ax_main)

        # Extract train/validation data points
        train_true, train_pred = true_values[is_train], pred_values[is_train]
        val_true, val_pred = true_values[~is_train], pred_values[~is_train]

        # Plot main scatter
        ax_main.scatter(train_true, train_pred, c='royalblue', alpha=0.7, label='Training')
        ax_main.scatter(val_true, val_pred, c='orange', alpha=0.8, label='Validation')

        # Calculate plot limits for parity plot
        # Determine if we use auto-calculated limits or user-provided limits
        if axlim is None:
            # Auto-calculate limits based on data
            data_min = min(np.min(true_values), np.min(pred_values))
            data_max = max(np.max(true_values), np.max(pred_values))
            data_range = data_max - data_min
            margin = 0.1 * data_range

            if log_scale:
                # For log scale, handle special cases with positive values
                limit_min = max(1e-10, data_min)  # Ensure minimum is positive
                if data_min <= 0:
                    limit_min = min(0.01 * data_max, 1e-6)  # Small positive value if data has zeros/negatives
                else:
                    limit_min = limit_min / 2  # Extend lower limit for better visualization

                limit_max = data_max * 5  # Extend upper limit
            else:
                # For linear scale, simply add margins
                limit_min = data_min - margin
                limit_max = data_max + margin
        else:
            # Use user-provided limits with appropriate margins
            if log_scale:
                # For log scale, extend limits logarithmically
                limit_min = axlim[0] / 5
                limit_max = axlim[1] * 10
            else:
                # For linear scale, add percentage-based margins
                range_size = axlim[1] - axlim[0]  # Note the corrected order
                limit_min = axlim[0] - 0.2 * range_size
                limit_max = axlim[1] + 0.2 * range_size

        # Add parity line and configure main plot
        ax_main.plot([limit_min, limit_max], [limit_min, limit_max], 'k--', label='Parity line')
        if log_scale:
            ax_main.set_xscale('log')
            ax_main.set_yscale('log')

        ax_main.grid(True, linestyle='--', alpha=0.5)
        ax_main.set_xlim(limit_min, limit_max)
        ax_main.set_ylim(limit_min, limit_max)
        ax_main.set_xlabel(xlabel, fontsize=self.lb_fsize)
        ax_main.set_ylabel(ylabel, fontsize=self.lb_fsize)

        # Prepare data for KDE plots
        dataframes = {
            'train_x': pd.DataFrame({"value": train_true}),
            'val_x': pd.DataFrame({"value": val_true}),
            'train_y': pd.DataFrame({"value": train_pred}),
            'val_y': pd.DataFrame({"value": val_pred})}

        # Configure KDE parameters based on scale type
        kde_kws = {'bw_adjust': 0.4 if kde_bandwidth is None else kde_bandwidth, 'fill': True, 'alpha': 0.4}
        if not log_scale:  # Only add clipping for linear scale
            kde_kws['clip'] = (data_min - margin / 2, data_max + margin / 2)

        # Plot marginal distributions
        sns.kdeplot(data=dataframes['train_x'], x="value", ax=ax_x_marginal, color='royalblue', **kde_kws)
        sns.kdeplot(data=dataframes['val_x'], x="value", ax=ax_x_marginal, color='orange', **kde_kws)
        sns.kdeplot(data=dataframes['train_y'], y="value", ax=ax_y_marginal, color='royalblue', **kde_kws)
        sns.kdeplot(data=dataframes['val_y'], y="value", ax=ax_y_marginal, color='orange', **kde_kws)

        # Clean up marginal plots (remove labels, ticks, and spines)
        for ax in [ax_x_marginal, ax_y_marginal]:
            ax.set_ylabel('')
            ax.set_xlabel('')
            ax.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
            for spine in ax.spines.values():
                spine.set_visible(False)

        # Add legend and metrics in title
        ax_main.legend(loc='upper left', fontsize=self.lg_fsize)

        title_with_metrics = (
            f'{title}\n'
            f'Training: R² = {train_metrics["R2"]:.3f}, RMSE = {train_metrics["RMSE"]:.3f} {unit}\n'
            f'Validation: R² = {val_metrics["R2"]:.3f}, RMSE = {val_metrics["RMSE"]:.3f} {unit}')

        fig.suptitle(title_with_metrics, fontsize=self.lb_fsize)

        # Finalize layout and save
        plt.tight_layout()
        fig.subplots_adjust(top=0.87, hspace=0.0, wspace=0.0)
        os.makedirs("./Plots", exist_ok=True)
        plt.savefig(f"./Plots/{filename}", dpi=self.dpi, bbox_inches='tight')
        plt.show()

        return fig

