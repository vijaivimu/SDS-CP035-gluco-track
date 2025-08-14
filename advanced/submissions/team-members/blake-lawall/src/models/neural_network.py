"""
Feedforward Neural Network for Diabetes Classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


class DiabetesFFNN(nn.Module):
    """
    Feedforward Neural Network for Diabetes Classification
    """
    
    def __init__(self, 
                 input_size: int,
                 hidden_sizes: List[int] = [128, 64, 32],
                 dropout_rate: float = 0.3,
                 activation: str = 'relu',
                 batch_norm: bool = True):
        """
        Initialize the neural network
        
        Args:
            input_size: Number of input features
            hidden_sizes: List of hidden layer sizes
            dropout_rate: Dropout rate for regularization
            activation: Activation function ('relu', 'tanh', 'sigmoid')
            batch_norm: Whether to use batch normalization
        """
        super(DiabetesFFNN, self).__init__()
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.dropout_rate = dropout_rate
        self.activation_name = activation
        self.batch_norm = batch_norm
        
        # Set activation function
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'tanh':
            self.activation = F.tanh
        elif activation == 'sigmoid':
            self.activation = F.sigmoid
        else:
            raise ValueError(f"Unsupported activation function: {activation}")
        
        # Build layers
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # Input layer
        prev_size = input_size
        
        # Hidden layers
        for hidden_size in hidden_sizes:
            self.layers.append(nn.Linear(prev_size, hidden_size))
            if batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(hidden_size))
            prev_size = hidden_size
        
        # Output layer
        self.output_layer = nn.Linear(prev_size, 1)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
            
        Returns:
            Output tensor of shape (batch_size, 1)
        """
        # Hidden layers
        for i, layer in enumerate(self.layers):
            x = layer(x)
            
            if self.batch_norm:
                x = self.batch_norms[i](x)
            
            x = self.activation(x)
            x = self.dropout(x)
        
        # Output layer
        x = self.output_layer(x)
        
        return x
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get probability predictions
        
        Args:
            x: Input tensor
            
        Returns:
            Probability predictions
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.sigmoid(logits)
        return probs
    
    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Get binary predictions
        
        Args:
            x: Input tensor
            threshold: Classification threshold
            
        Returns:
            Binary predictions
        """
        probs = self.predict_proba(x)
        return (probs >= threshold).float()


class DiabetesFFNNWithEmbeddings(nn.Module):
    """
    Feedforward Neural Network with Embeddings for Categorical Features
    """
    
    def __init__(self,
                 numerical_size: int,
                 categorical_sizes: List[int],
                 embedding_dims: List[int],
                 hidden_sizes: List[int] = [128, 64, 32],
                 dropout_rate: float = 0.3,
                 activation: str = 'relu',
                 batch_norm: bool = True):
        """
        Initialize the neural network with embeddings
        
        Args:
            numerical_size: Number of numerical features
            categorical_sizes: List of categorical feature cardinalities
            embedding_dims: List of embedding dimensions for each categorical feature
            hidden_sizes: List of hidden layer sizes
            dropout_rate: Dropout rate for regularization
            activation: Activation function
            batch_norm: Whether to use batch normalization
        """
        super(DiabetesFFNNWithEmbeddings, self).__init__()
        
        self.numerical_size = numerical_size
        self.categorical_sizes = categorical_sizes
        self.embedding_dims = embedding_dims
        self.hidden_sizes = hidden_sizes
        self.dropout_rate = dropout_rate
        self.activation_name = activation
        self.batch_norm = batch_norm
        
        # Set activation function
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'tanh':
            self.activation = F.tanh
        elif activation == 'sigmoid':
            self.activation = F.sigmoid
        else:
            raise ValueError(f"Unsupported activation function: {activation}")
        
        # Embedding layers
        self.embeddings = nn.ModuleList()
        for cat_size, emb_dim in zip(categorical_sizes, embedding_dims):
            self.embeddings.append(nn.Embedding(cat_size, emb_dim))
        
        # Calculate total input size
        total_embedding_size = sum(embedding_dims)
        input_size = numerical_size + total_embedding_size
        
        # Build layers
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # Hidden layers
        prev_size = input_size
        for hidden_size in hidden_sizes:
            self.layers.append(nn.Linear(prev_size, hidden_size))
            if batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(hidden_size))
            prev_size = hidden_size
        
        # Output layer
        self.output_layer = nn.Linear(prev_size, 1)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)
    
    def forward(self, 
                numerical_features: torch.Tensor,
                categorical_features: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through the network
        
        Args:
            numerical_features: Numerical features tensor
            categorical_features: List of categorical feature tensors
            
        Returns:
            Output tensor
        """
        # Process embeddings
        embedded_features = []
        for i, (embedding, cat_feat) in enumerate(zip(self.embeddings, categorical_features)):
            embedded = embedding(cat_feat.long())
            embedded_features.append(embedded)
        
        # Concatenate numerical and embedded features
        if embedded_features:
            embedded_concat = torch.cat(embedded_features, dim=1)
            x = torch.cat([numerical_features, embedded_concat], dim=1)
        else:
            x = numerical_features
        
        # Hidden layers
        for i, layer in enumerate(self.layers):
            x = layer(x)
            
            if self.batch_norm:
                x = self.batch_norms[i](x)
            
            x = self.activation(x)
            x = self.dropout(x)
        
        # Output layer
        x = self.output_layer(x)
        
        return x
    
    def predict_proba(self, 
                     numerical_features: torch.Tensor,
                     categorical_features: List[torch.Tensor]) -> torch.Tensor:
        """Get probability predictions"""
        self.eval()
        with torch.no_grad():
            logits = self.forward(numerical_features, categorical_features)
            probs = torch.sigmoid(logits)
        return probs
    
    def predict(self, 
                numerical_features: torch.Tensor,
                categorical_features: List[torch.Tensor],
                threshold: float = 0.5) -> torch.Tensor:
        """Get binary predictions"""
        probs = self.predict_proba(numerical_features, categorical_features)
        return (probs >= threshold).float()


def create_model(input_size: int,
                model_type: str = 'ffnn',
                hidden_sizes: List[int] = [128, 64, 32],
                dropout_rate: float = 0.3,
                activation: str = 'relu',
                batch_norm: bool = True,
                **kwargs) -> nn.Module:
    """
    Factory function to create neural network models
    
    Args:
        input_size: Number of input features
        model_type: Type of model ('ffnn' or 'ffnn_embeddings')
        hidden_sizes: List of hidden layer sizes
        dropout_rate: Dropout rate
        activation: Activation function
        batch_norm: Whether to use batch normalization
        **kwargs: Additional arguments for specific model types
        
    Returns:
        Neural network model
    """
    
    if model_type == 'ffnn':
        return DiabetesFFNN(
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            dropout_rate=dropout_rate,
            activation=activation,
            batch_norm=batch_norm
        )
    elif model_type == 'ffnn_embeddings':
        return DiabetesFFNNWithEmbeddings(
            numerical_size=kwargs.get('numerical_size', input_size),
            categorical_sizes=kwargs.get('categorical_sizes', []),
            embedding_dims=kwargs.get('embedding_dims', []),
            hidden_sizes=hidden_sizes,
            dropout_rate=dropout_rate,
            activation=activation,
            batch_norm=batch_norm
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}") 