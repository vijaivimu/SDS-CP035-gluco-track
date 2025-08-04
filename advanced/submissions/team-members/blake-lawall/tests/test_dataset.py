"""
Test for the dataset module
"""

import pytest
import pandas as pd
import numpy as np
import torch
from src.data.dataset import DiabetesDataset, create_data_loaders


class TestDiabetesDataset:
    """Test cases for DiabetesDataset"""
    
    def setup_method(self):
        """Set up test data"""
        # Create sample data
        np.random.seed(42)
        n_samples = 100
        
        self.sample_data = pd.DataFrame({
            'Diabetes_binary': np.random.randint(0, 2, n_samples),
            'Age': np.random.randint(18, 80, n_samples),
            'BMI': np.random.uniform(18, 40, n_samples),
            'GenHealth': np.random.choice(['Excellent', 'Good', 'Fair', 'Poor'], n_samples),
            'PhysicalActivity': np.random.randint(0, 2, n_samples),
            'Smoking': np.random.randint(0, 2, n_samples)
        })
    
    def test_dataset_creation(self):
        """Test dataset creation"""
        dataset = DiabetesDataset(self.sample_data)
        
        assert len(dataset) == 100
        assert dataset.get_input_size() > 0
        assert len(dataset.get_feature_names()) > 0
    
    def test_dataset_getitem(self):
        """Test getting items from dataset"""
        dataset = DiabetesDataset(self.sample_data)
        
        features, target = dataset[0]
        
        assert isinstance(features, torch.Tensor)
        assert isinstance(target, torch.Tensor)
        assert features.dim() == 1
        assert target.dim() == 1
        assert target.shape[0] == 1
    
    def test_data_loaders_creation(self):
        """Test data loaders creation"""
        train_loader, val_loader, test_loader, scaler, encoders = create_data_loaders(
            self.sample_data, batch_size=16
        )
        
        assert train_loader is not None
        assert val_loader is not None
        assert test_loader is not None
        
        # Test that we can iterate through the loaders
        for batch_features, batch_targets in train_loader:
            assert batch_features.shape[0] <= 16  # batch_size
            assert batch_targets.shape[0] <= 16
            break


if __name__ == "__main__":
    pytest.main([__file__]) 