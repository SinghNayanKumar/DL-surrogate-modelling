import torch
import torch.nn.functional as F
from tqdm import tqdm
from training.physics_loss import pde_loss

class ModelTrainer:
    def __init__(self, model, train_loader, val_loader, optimizer, device, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device
        self.config = config
        self.model_type = config['model_type']
        self.use_pinn = config['use_pinn']
        self.pinn_weight = config.get('pinn_weight', 1e-6)

    def _run_epoch(self, loader, is_train=True):
        self.model.train() if is_train else self.model.eval()
        total_loss, total_data_loss, total_pde_loss = 0, 0, 0
        
        for data in tqdm(loader, desc="Training" if is_train else "Validation"):
            if self.model_type == 'unet':
                inputs, targets = data
                inputs, targets = inputs.to(self.device), targets.to(self.device)
            else: # GNNs
                data = data.to(self.device)
                inputs, targets = data, data.y
            
            if is_train: self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(inputs)
            
            # Data loss (always computed)
            data_loss = F.mse_loss(outputs, targets)
            loss = data_loss
            
            # PDE loss (only during training and if enabled)
            pde_loss_val = torch.tensor(0.0)
            if is_train and self.use_pinn and self.model_type != 'unet':
                # PINN loss requires coordinates to have grad enabled
                coords = data.x.clone().detach().requires_grad_(True)
                displacement = self.model(Data(x=coords, edge_index=data.edge_index))
                
                # Note: For simplicity, E and ν are hardcoded. In a real scenario,
                # you would pass them from the dataset's global attributes.
                pde_loss_val = pde_loss(displacement, coords)
                loss = data_loss + self.pinn_weight * pde_loss_val

            if is_train:
                loss.backward()
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
            train_loss, train_data_loss, train_pde_loss = self._run_epoch(self.train_loader, is_train=True)
            print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.6f} | Data Loss: {train_data_loss:.6f} | PDE Loss: {train_pde_loss:.6f}")

            val_loss, val_data_loss, _ = self._run_epoch(self.val_loader, is_train=False)
            print(f"Epoch {epoch:03d} | Val Loss:   {val_loss:.6f} | Val Data Loss: {val_data_loss:.6f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_path = f"best_model_{self.config['experiment_name']}.pth"
                torch.save(self.model.state_dict(), save_path)
                print(f"  -> New best model saved to {save_path}")