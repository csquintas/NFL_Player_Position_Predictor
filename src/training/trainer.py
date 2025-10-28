import torch
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime

class Trainer:
    """Manage training and evaluation loops"""

    def __init__(self, cfg, optim, criterion, run_desc=None):
        """Initialize a trainer instance tied to a specific experiment setup.

        Args:
            cfg: Training configuration TrainConfig()
            optim: Optimizer instance used to update model parameters.
            criterion: Loss function applied to model predictions versus targets.
            run_desc: Optional short description for run folder.
        """
        self.epochs_completed = 0

        self.logdir = os.path.join("runs", f"{datetime.now().strftime("%Y%m%d_%H%M%S")}" + (f"_{run_desc}" if run_desc else ""))
        
        self.writer = SummaryWriter(self.logdir)

        self.cfg = cfg
        self.optim = optim
        self.criterion = criterion
        self.ckpt_path = None
        if self.cfg.checkpoint:
            self.ckpt_path = os.path.join(self.logdir, "checkpoints")
            os.makedirs(self.ckpt_path, exist_ok=True)

    def fit(self, model, train_loader, val_loader=None):
        """Training model code with ability for valuation

        Args:
            model: Torch module implementing `forward(hist, mask, last_known_xy)`.
            train_loader: Iterable yielding batches of training tensors and metadata.
            val_loader: Optional iterable for validation; when provided best checkpoints
                are tracked using validation RMSE.
        """

        device = self.cfg.device
        num_epochs = self.cfg.num_epochs
        batch_sz = self.cfg.batch_size

        model.to(device)
        global_step = 0
        best_train_rmse, best_val_rmse = float("inf"), float("inf")
        for epoch in range(1, num_epochs + 1):
            model.train()
            epoch_train_loss_sum, n_samples, n_valid_timesteps = 0.0, 0, 0

            iter_bar = tqdm(train_loader, desc=f"Train epoch {epoch}/{num_epochs}", leave=False)
            for iters, (hist, padded_frame_mask, last_known_xy, target_xy, meta) in enumerate(iter_bar, 1):
                hist, padded_frame_mask, last_known_xy, target_xy = [t.to(device) for t in (hist, padded_frame_mask, last_known_xy, target_xy)]

                preds = model(hist, padded_frame_mask, last_known_xy)

                T_use = target_xy.size(1)
                preds = preds[:, :T_use, :] 
                
                elem_loss = self.criterion(preds, target_xy)

                per_step = elem_loss.mean(dim=-1)                       # [B, T_max]

                # mask out padded predictions
                target_mask = meta["target_mask"].to(per_step.device)   # [B, T_max]
                loss = per_step[target_mask].mean()

                self.optim.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
                self.optim.step()


                valid_count = target_mask.sum().item()                      # # of valid timesteps this batch
                epoch_train_loss_sum += per_step[target_mask].sum().item()  # sum of per-timestep MSE
                n_valid_timesteps     += valid_count                       

                iter_bar.set_postfix(loss=f"{loss.item():.4f}")

                self.writer.add_scalar("train/step_loss", loss.item(), global_step)
                self.writer.add_scalar("train/grad_norm", grad_norm, global_step)

                global_step += 1
        
            self.epochs_completed += 1

            if val_loader:
                val_mse, val_rmse = self.evaluate(model, val_loader)
            else:
                val_mse, val_rmse = None, None

            train_mse  = epoch_train_loss_sum / max(n_valid_timesteps, 1)
            train_rmse = train_mse ** 0.5

            self.writer.add_scalar("train/epoch_mse", train_mse, epoch)
            self.writer.add_scalar("train/epoch_rmse", train_rmse, epoch)

            ckpt_path = os.path.join(self.ckpt_path, f"epoch_{epoch}.pt")
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": self.optim.state_dict(),
                "train_rmse": train_rmse,
                "val_rmse": val_rmse,
                "config": self.cfg.__dict__ if hasattr(self.cfg, "__dict__") else dict(self.cfg),
            }, ckpt_path)


            if val_rmse is not None and val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                best_path = os.path.join(self.ckpt_path, "best_model_validation.pt")
                torch.save({
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": self.optim.state_dict(),
                    "val_rmse": val_rmse,
                }, best_path)
                print(f"Saved best model at epoch {epoch} (val rmse={val_rmse:.4f})")

            if train_rmse is not None and train_rmse < best_train_rmse:
                best_train_rmse = train_rmse
                best_path = os.path.join(self.ckpt_path, "best_model_train.pt")
                torch.save({
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": self.optim.state_dict(),
                    "train_rmse": train_rmse,
                }, best_path)
                print(f"Saved best model at epoch {epoch} (train rmse={train_rmse:.4f})")
            

    
    def evaluate(self, model, val_loader):
        """Run the model in evaluation mode and compute validation metrics.

        Args:
            model: Module to be evaluated.
            val_loader: Data loader yielding validation batches.

        Returns:
            Tuple containing mean squared error and root mean squared error.
        """
        device = self.cfg.device
        model.to(device)
        batch_sz = self.cfg.batch_size
        model.eval()

        epoch_val_loss_sum, n_samples, n_valid_timesteps = 0.0, 0 ,0

        with torch.no_grad():
            iter_bar = tqdm(val_loader, desc="Evaluation", leave=False)
            for hist, padded_frame_mask, last_known_xy, target_xy, meta in iter_bar:
                hist, padded_frame_mask, last_known_xy, target_xy = [t.to(device) for t in (hist, padded_frame_mask, last_known_xy, target_xy)]

                preds = model(hist, padded_frame_mask, last_known_xy)
                T_use = target_xy.size(1)
                preds = preds[:, :T_use, :] # mask out to longest in the batch

                #loss = self.criterion(preds, target_xy)
                elem_loss = self.criterion(preds, target_xy)

                per_step = elem_loss.mean(dim=-1)                       # [B, T_max]

                # mask out padded predictions
                target_mask = meta["target_mask"].to(per_step.device)   # [B, T_max]
                loss = per_step[target_mask].mean()
                
                valid_count = target_mask.sum().item()                     # # of valid timesteps this batch
                epoch_val_loss_sum += per_step[target_mask].sum().item()   # sum of per-timestep MSE
                n_valid_timesteps     += valid_count                      

            val_mse = epoch_val_loss_sum / max(n_valid_timesteps, 1)
            val_rmse = val_mse ** 0.5
            self.writer.add_scalar("val/epoch_mse", val_mse, self.epochs_completed)
            self.writer.add_scalar("val/epoch_rmse", val_rmse, self.epochs_completed)

        return val_mse, val_rmse
