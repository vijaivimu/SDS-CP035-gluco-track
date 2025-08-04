"""
Training utilities for neural network models
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
import mlflow.pytorch
from typing import Dict, List, Optional, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os


class EarlyStopping:
    """Early stopping utility to prevent overfitting"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_weights = model.state_dict().copy()
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False


class ModelTrainer:
    """Trainer class for neural network models"""
    
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 test_loader: Optional[DataLoader] = None,
                 learning_rate: float = 0.001,
                 weight_decay: float = 1e-5,
                 device: str = 'auto'):
        """
        Initialize the trainer
        
        Args:
            model: Neural network model
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader (optional)
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for regularization
            device: Device to use ('auto', 'cpu', 'cuda')
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        
        # Setup optimizer and loss function
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_metrics = []
        self.val_metrics = []
        
    def train_epoch(self) -> Tuple[float, Dict[str, float]]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Get predictions for metrics
            probs = torch.sigmoid(output).cpu().detach().numpy()
            preds = (probs >= 0.5).astype(int)
            targets = target.cpu().detach().numpy()
            
            all_predictions.extend(preds.flatten())
            all_targets.extend(targets.flatten())
        
        avg_loss = total_loss / len(self.train_loader)
        metrics = self._calculate_metrics(all_targets, all_predictions)
        
        return avg_loss, metrics
    
    def validate(self, loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        all_probs = []
        
        with torch.no_grad():
            for data, target in loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                total_loss += loss.item()
                
                probs = torch.sigmoid(output).cpu().numpy()
                preds = (probs >= 0.5).astype(int)
                targets = target.cpu().numpy()
                
                all_probs.extend(probs.flatten())
                all_predictions.extend(preds.flatten())
                all_targets.extend(targets.flatten())
        
        avg_loss = total_loss / len(loader)
        metrics = self._calculate_metrics(all_targets, all_predictions, all_probs)
        
        return avg_loss, metrics
    
    def _calculate_metrics(self, 
                          targets: List[int], 
                          predictions: List[int], 
                          probabilities: Optional[List[float]] = None) -> Dict[str, float]:
        """Calculate evaluation metrics"""
        metrics = {
            'accuracy': accuracy_score(targets, predictions),
            'precision': precision_score(targets, predictions, zero_division=0),
            'recall': recall_score(targets, predictions, zero_division=0),
            'f1': f1_score(targets, predictions, zero_division=0)
        }
        
        if probabilities is not None:
            metrics['auc'] = roc_auc_score(targets, probabilities)
        
        return metrics
    
    def train(self,
              epochs: int,
              early_stopping_patience: int = 10,
              save_path: Optional[str] = None,
              mlflow_experiment: Optional[str] = None) -> Dict[str, List[float]]:
        """
        Train the model
        
        Args:
            epochs: Number of training epochs
            early_stopping_patience: Patience for early stopping
            save_path: Path to save the best model
            mlflow_experiment: MLflow experiment name
            
        Returns:
            Training history
        """
        early_stopping = EarlyStopping(patience=early_stopping_patience)
        
        # Setup MLflow if specified
        if mlflow_experiment:
            mlflow.set_experiment(mlflow_experiment)
            mlflow.start_run()
            
            # Log model parameters
            mlflow.log_params({
                'learning_rate': self.optimizer.param_groups[0]['lr'],
                'weight_decay': self.optimizer.param_groups[0]['weight_decay'],
                'epochs': epochs,
                'early_stopping_patience': early_stopping_patience
            })
        
        print(f"Training on device: {self.device}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        
        for epoch in range(epochs):
            # Training
            train_loss, train_metrics = self.train_epoch()
            
            # Validation
            val_loss, val_metrics = self.validate(self.val_loader)
            
            # Store history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_metrics.append(train_metrics)
            self.val_metrics.append(val_metrics)
            
            # Print progress
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"Train Loss: {train_loss:.4f}, Train F1: {train_metrics['f1']:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val F1: {val_metrics['f1']:.4f}")
            print("-" * 50)
            
            # Log to MLflow
            if mlflow_experiment:
                mlflow.log_metrics({
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'train_accuracy': train_metrics['accuracy'],
                    'val_accuracy': val_metrics['accuracy'],
                    'train_f1': train_metrics['f1'],
                    'val_f1': val_metrics['f1'],
                    'train_auc': train_metrics.get('auc', 0),
                    'val_auc': val_metrics.get('auc', 0)
                }, step=epoch)
            
            # Early stopping
            if early_stopping(val_loss, self.model):
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        # Save best model
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(self.model.state_dict(), save_path)
            print(f"Best model saved to {save_path}")
        
        # Log model to MLflow
        if mlflow_experiment:
            mlflow.pytorch.log_model(self.model, "model")
            mlflow.end_run()
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics
        }
    
    def evaluate(self, loader: Optional[DataLoader] = None) -> Dict[str, float]:
        """Evaluate the model on test set"""
        if loader is None:
            if self.test_loader is None:
                raise ValueError("No test loader provided")
            loader = self.test_loader
        
        loss, metrics = self.validate(loader)
        
        print("Test Results:")
        for metric, value in metrics.items():
            print(f"{metric.capitalize()}: {value:.4f}")
        
        return metrics
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """Plot training history"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plot
        axes[0, 0].plot(self.train_losses, label='Train Loss')
        axes[0, 0].plot(self.val_losses, label='Validation Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy plot
        train_acc = [m['accuracy'] for m in self.train_metrics]
        val_acc = [m['accuracy'] for m in self.val_metrics]
        axes[0, 1].plot(train_acc, label='Train Accuracy')
        axes[0, 1].plot(val_acc, label='Validation Accuracy')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # F1 Score plot
        train_f1 = [m['f1'] for m in self.train_metrics]
        val_f1 = [m['f1'] for m in self.val_metrics]
        axes[1, 0].plot(train_f1, label='Train F1')
        axes[1, 0].plot(val_f1, label='Validation F1')
        axes[1, 0].set_title('Training and Validation F1 Score')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # AUC plot (if available)
        if 'auc' in self.train_metrics[0]:
            train_auc = [m['auc'] for m in self.train_metrics]
            val_auc = [m['auc'] for m in self.val_metrics]
            axes[1, 1].plot(train_auc, label='Train AUC')
            axes[1, 1].plot(val_auc, label='Validation AUC')
            axes[1, 1].set_title('Training and Validation AUC')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('AUC')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training history plot saved to {save_path}")
        
        plt.show()
    
    def save_model(self, path: str):
        """Save the model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load a saved model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.train_metrics = checkpoint.get('train_metrics', [])
        self.val_metrics = checkpoint.get('val_metrics', [])
        print(f"Model loaded from {path}") 