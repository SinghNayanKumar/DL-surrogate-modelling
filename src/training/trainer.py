import torch
import torch.nn.functional as F
from tqdm import tqdm
import wandb
from src.training.physics_loss import pde_loss
from torch_geometric.data import Data 
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
        self.pinn_weight = config.get('pinn_weight', 1e-6)

    def _run_epoch(self, loader, is_train=True):
        self.model.train() if is_train else self.model.eval()
        total_loss, total_data_loss, total_pde_loss = 0, 0, 0
        
        # ### --- NOTE --- ### The `desc` provides a descriptive label for the progress bar.
        for data in tqdm(loader, desc=f"Epoch {self.epoch:03d} - {'Train' if is_train else 'Valid'}"):
            if self.model_type.startswith('unet'):
                inputs, targets = data
                inputs, targets = inputs.to(self.device), targets.to(self.device)
            else: # GNNs
                data = data.to(self.device)
                # For GNNs, the `inputs` are the entire graph object, and `targets` are a specific attribute.
                inputs, targets = data, data.y
            
            if is_train: self.optimizer.zero_grad(set_to_none=True) # Use set_to_none=True for slight perf gain
            
            
            # The entire PINN logic is refactored for correct gradient flow.
            
            # --- Step 1: Prepare inputs and enable gradients for PINN ---
            pde_loss_val = torch.tensor(0.0, device=self.device, dtype=torch.float32)
            if self.use_pinn and self.model_type != 'unet':
                # For PINN, the input coordinates MUST have gradients enabled to compute derivatives.
                # We clone `data.pos` which contains the original coordinates.
                pinn_inputs = data.clone()
                pinn_inputs.x = pinn_inputs.x.to(torch.float32)
                pinn_inputs.pos = pinn_inputs.pos.to(torch.float32).requires_grad_(True)
                # The model's forward pass now receives the graph with grad-enabled positions
                outputs = self.model(pinn_inputs)
            else:
                # Standard forward pass for data-driven mode or U-Net
                outputs = self.model(inputs)

            # --- Step 2: Compute Data Loss (always) ---
            # This loss compares the model's prediction to the ground truth from the simulation.
            data_loss = F.mse_loss(outputs, targets)
            loss = data_loss
            
            # --- Step 3: Compute PDE Loss (if PINN is enabled) ---
            if is_train and self.use_pinn and self.model_type != 'unet':
                # The `pde_loss` function uses the `outputs` from the SINGLE forward pass
                # and the grad-enabled `pinn_inputs.pos` to calculate the physical residual.
                
                # Unpack material properties for the current batch
                youngs_modulus = data.material_props[:, 0].to(torch.float32)
                poissons_ratio = data.material_props[:, 1].to(torch.float32)
                
                # The graph batching in PyG needs to be handled. We need to pass the batch index.
                pde_loss_val = pde_loss(outputs, pinn_inputs.pos, youngs_modulus, poissons_ratio, batch=pinn_inputs.batch)
                
                # Add the weighted PDE loss to the total loss
                loss = data_loss + self.pinn_weight * pde_loss_val

            # --- Step 4: Backward pass and optimization ---
            if is_train:
                loss.backward() # Gradients from both data_loss and pde_loss flow back
                self.optimizer.step()
            
            batch_size = data.num_graphs if hasattr(data, 'num_graphs') else inputs.size(0)
            total_loss += loss.item() * batch_size
            total_data_loss += data_loss.item() * batch_size
            total_pde_loss += pde_loss_val.item() * batch_size

        num_samples = len(loader.dataset)
        avg_loss = total_loss / num_samples
        avg_data_loss = total_data_loss / num_samples
        avg_pde_loss = total_pde_loss / num_samples
        return avg_loss, avg_data_loss, avg_pde_loss

    def train(self, num_epochs):
        best_val_loss = float('inf')
        for epoch in range(1, num_epochs + 1):
            self.epoch = epoch # Store epoch for progress bar description

            # --- Training Step ---
            train_loss, train_data_loss, train_pde_loss = self._run_epoch(self.train_loader, is_train=True)
            print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.6f} | Data Loss: {train_data_loss:.6f} | PDE Loss: {train_pde_loss:.6f}")
            
            # Validation is always done without the PDE loss term for a fair comparison of data-fit.
            with torch.no_grad():
                val_loss, val_data_loss, _ = self._run_epoch(self.val_loader, is_train=False)
            print(f"Epoch {epoch:03d} | Val Loss:   {val_loss:.6f} | Val Data Loss: {val_data_loss:.6f}")

            # ### --- W&B INTEGRATION --- ###
            # Log all metrics to W&B for this epoch
            wandb.log({
                "epoch": epoch,
                "train/total_loss": train_loss,
                "train/data_loss": train_data_loss,
                "train/pde_loss": train_pde_loss,
                "val/total_loss": val_loss,
                "val/data_loss": val_data_loss,
                # We can also track learning rate if using a scheduler
                "learning_rate": self.optimizer.param_groups[0]['lr'] 
            })
            # ### --- END W&B INTEGRATION --- ###

            # ### --- SCHEDULER STEP --- ###
            # Step the scheduler based on the validation loss
            if self.scheduler:
                self.scheduler.step(val_loss)

            # --- Model Checkpointing ---
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # The save path now uses the name from trainer_config, which is the W&B run name
                save_path = f"checkpoints/best_model_{self.config['experiment_name']}.pth"
                
                # Create the checkpoints directory if it doesn't exist
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                
                torch.save(self.model.state_dict(), save_path)
                print(f"  -> New best model saved to {save_path}")
                
                # ### --- W&B INTEGRATION --- ###
                # Update the summary metrics in W&B to reflect the best score
                wandb.summary['best_val_loss'] = best_val_loss
                wandb.summary['best_epoch'] = epoch