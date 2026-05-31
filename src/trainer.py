"""
Universal Trainer for TimeXer, iTransformer, TFT
Handles training loop, validation, early stopping, checkpointing
"""
import os, time, json, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR


def seed_everything(seed=42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


class EarlyStopping:
    def __init__(self, patience=7, min_delta=1e-5):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = np.inf
        self.best_state = None

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
        return self.counter >= self.patience


class DeepTrainer:
    def __init__(self, model, config, device=None):
        self.model  = model
        self.config = config
        self.device = device or torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        self.optimizer = AdamW(
            model.parameters(),
            lr=config.get('lr', 1e-4),
            weight_decay=config.get('weight_decay', 1e-4)
        )
        self.criterion = nn.HuberLoss(delta=1.0)  # robust to outliers
        self.history = {'train_loss': [], 'val_loss': [], 'val_mae': []}

    def train_epoch(self, loader):
        self.model.train()
        losses = []
        for batch in loader:
            x_enc    = batch['x_enc'].to(self.device)
            x_exog   = batch['x_exog'].to(self.device)
            x_future = batch['x_future'].to(self.device)
            x_mark   = batch['x_mark'].to(self.device)
            y        = batch['y_norm'].to(self.device)

            self.optimizer.zero_grad()
            pred = self.model(x_enc, x_exog, x_future, x_mark)
            loss = self.criterion(pred, y)
            # L1 sparsity regularization on module scale parameters
            if hasattr(self.model, 'scale_l1_loss'):
                scale_l1_lambda = self.config.get('scale_l1_lambda', 0.0)
                if scale_l1_lambda > 0:
                    loss = loss + scale_l1_lambda * self.model.scale_l1_loss()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            losses.append(loss.item())
        return np.mean(losses)

    @torch.no_grad()
    def eval_epoch(self, loader, scaler_stats):
        self.model.eval()
        all_preds, all_targets = [], []
        val_losses = []

        endog_mean = scaler_stats['endog_mean'][0]
        endog_std  = scaler_stats['endog_std'][0]

        for batch in loader:
            x_enc    = batch['x_enc'].to(self.device)
            x_exog   = batch['x_exog'].to(self.device)
            x_future = batch['x_future'].to(self.device)
            x_mark   = batch['x_mark'].to(self.device)
            y_norm   = batch['y_norm'].to(self.device)
            y_raw    = batch['y_raw']

            pred_norm = self.model(x_enc, x_exog, x_future, x_mark)
            loss = self.criterion(pred_norm, y_norm)
            val_losses.append(loss.item())

            # Denormalize predictions
            pred_gw = pred_norm.cpu().numpy() * endog_std + endog_mean
            all_preds.append(pred_gw)
            all_targets.append(y_raw.numpy())

        all_preds   = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        mae  = np.mean(np.abs(all_preds - all_targets))
        rmse = np.sqrt(np.mean((all_preds - all_targets) ** 2))
        return np.mean(val_losses), mae, rmse, all_preds, all_targets

    def fit(self, train_dataset, val_dataset, model_name='model', log_dir='/home/csh/myproject/logs', seed=42):
        config = self.config
        batch_size = config.get('batch_size', 32)
        max_epochs = config.get('max_epochs', 30)
        patience   = config.get('patience', 7)

        # Set seed for reproducibility
        seed_everything(seed)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  num_workers=4, pin_memory=True, drop_last=True)
        val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False,
                                  num_workers=4, pin_memory=True)

        scheduler = CosineAnnealingLR(self.optimizer, T_max=max_epochs, eta_min=1e-6)
        stopper   = EarlyStopping(patience=patience)

        print(f"\n[Trainer] Starting training: {model_name} (seed={seed})")
        print(f"  Device: {self.device}  |  Train: {len(train_dataset)} samples  |  Val: {len(val_dataset)} samples")
        print(f"  Batch: {batch_size}  |  Max epochs: {max_epochs}  |  Patience: {patience}")

        for epoch in range(1, max_epochs + 1):
            t0 = time.time()
            train_loss = float(self.train_epoch(train_loader))
            val_loss, val_mae, val_rmse, _, _ = self.eval_epoch(val_loader, val_dataset.scaler_stats)
            val_loss = float(val_loss)
            val_mae = float(val_mae)
            val_rmse = float(val_rmse)
            scheduler.step()

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_mae'].append(val_mae)

            elapsed = time.time() - t0
            print(f"  Epoch {epoch:3d}/{max_epochs} | "
                  f"TrainLoss={train_loss:.4f} | ValLoss={val_loss:.4f} | "
                  f"ValMAE={val_mae:.4f} GW | {elapsed:.1f}s")

            # Log progress
            log_path = os.path.join(log_dir, f'{model_name}_progress.json')
            os.makedirs(log_dir, exist_ok=True)
            with open(log_path, 'w') as f:
                json.dump({'epoch': epoch, 'train_loss': train_loss,
                           'val_loss': val_loss, 'val_mae': val_mae,
                           'history': self.history}, f)

            if stopper(val_loss, self.model):
                print(f"  Early stopping at epoch {epoch}. Best val_loss={stopper.best_loss:.4f}")
                break

        # Restore best weights
        if stopper.best_state is not None:
            self.model.load_state_dict(stopper.best_state)
            self.model.to(self.device)

        return self.history

    @torch.no_grad()
    def predict(self, test_dataset, scaler_stats):
        self.model.eval()
        loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)
        all_preds, all_targets = [], []

        endog_mean = scaler_stats['endog_mean'][0]
        endog_std  = scaler_stats['endog_std'][0]

        for batch in loader:
            x_enc    = batch['x_enc'].to(self.device)
            x_exog   = batch['x_exog'].to(self.device)
            x_future = batch['x_future'].to(self.device)
            x_mark   = batch['x_mark'].to(self.device)
            y_raw    = batch['y_raw']

            pred_norm = self.model(x_enc, x_exog, x_future, x_mark)
            pred_gw   = pred_norm.cpu().numpy() * endog_std + endog_mean
            all_preds.append(pred_gw)
            all_targets.append(y_raw.numpy())

        return np.concatenate(all_preds), np.concatenate(all_targets)
