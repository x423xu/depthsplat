import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
# import distributed sampler
from torch.utils.data.distributed import DistributedSampler

# ----------------------------
# 1. Data Module (LightningDataModule)
# ----------------------------
class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir='./data', batch_size=64):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    def prepare_data(self):
        # Download dataset (called only on 1 GPU in DDP)
        datasets.MNIST(self.data_dir, train=True, download=True)
        datasets.MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        # Split dataset (called on every GPU)
        full_dataset = datasets.MNIST(self.data_dir, train=True, transform=self.transform)
        self.train_dataset, self.val_dataset = random_split(full_dataset, [55000, 5000])
        self.test_dataset = datasets.MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=4)

# ----------------------------
# 2. Model (LightningModule)
# ----------------------------
class LitModel(pl.LightningModule):
    def __init__(self, lr=0.001):
        super().__init__()
        self.lr = lr
        self.l1 = nn.Linear(28 * 28, 128)
        self.l2 = nn.Linear(128, 10)
        self.save_hyperparameters()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.l1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.l2(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss, sync_dist=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('val_loss', loss, sync_dist=True, prog_bar=True)
        
        # Calculate accuracy
        preds = torch.argmax(y_hat, dim=1)
        acc = (preds == y).float().mean()
        self.log('val_acc', acc, sync_dist=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('test_loss', loss, sync_dist=True, prog_bar=True)
        
        # Calculate accuracy
        preds = torch.argmax(y_hat, dim=1)
        acc = (preds == y).float().mean()
        self.log('test_acc', acc, sync_dist=True, prog_bar=True)
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

# ----------------------------
# 3. Main Training Script
# ----------------------------
if __name__ == '__main__':
    # Configuration
    config = {
        'max_epochs': 5,
        'batch_size': 128,
        'accelerator': 'gpu' if torch.cuda.is_available() else 'auto',
        'devices': torch.cuda.device_count() if torch.cuda.is_available() else None,
        'strategy': DDPStrategy(find_unused_parameters=False) if torch.cuda.is_available() else None,
        'log_every_n_steps': 10,
        'precision': 16  # Mixed precision training
    }
    
    # Initialize components
    data_module = MNISTDataModule(batch_size=config['batch_size'])
    model = LitModel(lr=0.001)
    
    # Trainer configuration
    trainer = pl.Trainer(
        max_epochs=config['max_epochs'],
        accelerator=config['accelerator'],
        devices=config['devices'],
        strategy=config['strategy'],
        # strategy = DDPStrategy(  # Explicit DDP with Gloo [1](@ref)
        #                                     cluster_environment=None,  # Default uses Gloo
        #                                     process_group_backend="gloo"  # Force Gloo backend
                                        # ),
        log_every_n_steps=config['log_every_n_steps'],
        precision=config['precision'],
        enable_progress_bar=True,
        enable_model_summary=True
    )
    
    # Start training
    trainer.fit(model, datamodule=data_module)
    
    # Test after training
    if trainer.is_global_zero:  # Only run on root process
        trainer.test(model, datamodule=data_module)