# --- START OF FILE src/training/trainer.py ---
import torch
import torch.nn.functional as F
from tqdm import tqdm
import wandb
from src.training.physics_loss import pde_loss
import os

class ModelTrainer:
    def __init__(self, model, train_loader, val_loader, optimizer, device, config, scheduler=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config
        self.model_type = config['model_type']
        self.use_pinn = config['use_pinn']
        self.stats = config.get('stats', None)
        if self.use_pinn and self.stats is None:
            raise ValueError("Normalization stats must be provided to the trainer for PINN loss calculation.")

        # --- NEW: Curriculum Learning Parameters ---
        self.pinn_weight_final = config.get('pinn_weight', 1e-6)
        self.pinn_warmup_epochs = config.get('pinn_warmup_epochs', 25) # Default to 25 epochs of warmup
        self.current_pinn_weight = 0.0 # Start with zero physics loss

    def _update_pinn_weight(self):
        """Linearly increase the PINN weight during the warmup phase."""
        if self.epoch <= self.pinn_warmup_epochs:
            self.current_pinn_weight = self.pinn_weight_final * (self.epoch / self.pinn_warmup_epochs)
        else:
            self.current_pinn_weight = self.pinn_weight_final
        
        # Log the dynamic weight to wandb to see it change
        wandb.log({"epoch": self.epoch, "train/pinn_weight_dynamic": self.current_pinn_weight})

    def _run_epoch(self, loader, is_train=True):
        if is_train:
            self.model.train()
            if self.use_pinn:
                self._update_pinn_weight() # Update weight at the start of each training epoch
        else:
            self.model.eval()

        total_loss, total_data_loss, total_pde_loss = 0, 0, 0
        
        for data in tqdm(loader, desc=f"Epoch {self.epoch:03d} - {'Train' if is_train else 'Valid'}"):
            data = data.to(self.device)
            inputs, targets = data, data.y
            
            if is_train: self.optimizer.zero_grad(set_to_none=True)

            pinn_enabled_this_step = is_train and self.use_pinn and self.current_pinn_weight > 0

            if pinn_enabled_this_step:
                mean_x_coords = self.stats['mean_x'][:3].to(self.device)
                std_x_coords = self.stats['std_x'][:3].to(self.device)
                physical_coords = (data.pos * std_x_coords + mean_x_coords).detach().requires_grad_(True)
                normalized_coords_for_model = (physical_coords - mean_x_coords) / std_x_coords
                
                pinn_inputs = data.clone()
                pinn_inputs.pos = normalized_coords_for_model
                pinn_inputs.x = torch.cat([normalized_coords_for_model, data.x[:, 3:]], dim=1)
                
                outputs = self.model(pinn_inputs)
                data_loss = F.mse_loss(outputs, targets)
                
                mean_y = self.stats['mean_y'].to(self.device)
                std_y = self.stats['std_y'].to(self.device)
                physical_outputs = outputs * std_y + mean_y
                
                youngs_modulus = data.material_props[:, 0].to(torch.float32)
                poissons_ratio = data.material_props[:, 1].to(torch.float32)
                
                raw_pde_loss = pde_loss(physical_outputs, physical_coords, youngs_modulus, poissons_ratio, batch=pinn_inputs.batch)
                pde_loss_step = self.current_pinn_weight * raw_pde_loss
            else:
                with torch.set_grad_enabled(is_train):
                    outputs = self.model(inputs)
                    data_loss = F.mse_loss(outputs, targets)
                pde_loss_step = torch.tensor(0.0, device=self.device)

            loss = data_loss + pde_loss_step

            if is_train:
                loss.backward()
                self.optimizer.step()
            
            batch_size = data.num_graphs
            total_loss += loss.item() * batch_size
            total_data_loss += data_loss.item() * batch_size
            total_pde_loss += pde_loss_step.item() * batch_size
         
        num_samples = len(loader.dataset)
        epsilon = 1e-9
        avg_loss = total_loss / (num_samples + epsilon)
        avg_data_loss = total_data_loss / (num_samples + epsilon)
        avg_pde_loss = total_pde_loss / (num_samples + epsilon)
        return avg_loss, avg_data_loss, avg_pde_loss

    def train(self, num_epochs):
        best_val_loss = float('inf')
        for epoch in range(1, num_epochs + 1):
            self.epoch = epoch

            train_loss, train_data_loss, train_pde_loss = self._run_epoch(self.train_loader, is_train=True)
            print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.6f} | Data Loss: {train_data_loss:.6f} | PDE Loss: {train_pde_loss:.6f}")
            
            with torch.no_grad():
                val_loss, val_data_loss, _ = self._run_epoch(self.val_loader, is_train=False)
            print(f"Epoch {epoch:03d} | Val Loss:   {val_loss:.6f} | Val Data Loss: {val_data_loss:.6f}")

            wandb.log({
                "epoch": epoch, "train/total_loss": train_loss, "train/data_loss": train_data_loss,
                "train/pde_loss": train_pde_loss, "val/total_loss": val_loss, "val/data_loss": val_data_loss,
                "learning_rate": self.optimizer.param_groups[0]['lr'] 
            })

            if self.scheduler: self.scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_path = f"checkpoints/best_model_{self.config['experiment_name']}.pth"
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(self.model.state_dict(), save_path)
                print(f"  -> New best model saved to {save_path}")
                wandb.summary['best_val_loss'] = best_val_loss
                wandb.summary['best_epoch'] = epoch
# --- END OF FILE ---