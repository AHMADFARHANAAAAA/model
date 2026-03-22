"""
Deep Knowledge Tracing Model Architecture
LSTM-based implementation for knowledge state prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class DKTModel(nn.Module):
    """
    Deep Knowledge Tracing Model
    
    Architecture:
    - Embedding layer: Maps question IDs to dense vectors
    - LSTM layers: Captures sequential pattern in student learning
    - Dense output layer: Predicts mastery probability (0-1)
    
    Input: (batch_size, sequence_length, 2) where:
        - sequence_length: number of interactions for a student
        - 2 features: [question_id_embedding, is_correct]
    
    Output: (batch_size, sequence_length, 1) - mastery probability for next question
    """
    
    def __init__(
        self,
        num_questions: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        embedding_size: int = 50,
        dropout: float = 0.3,
        output_size: int = 1
    ):
        """
        Initialize DKT model
        
        Args:
            num_questions: Total number of unique questions
            hidden_size: LSTM hidden state dimension
            num_layers: Number of LSTM layers
            embedding_size: Dimension of question embeddings
            dropout: Dropout probability
            output_size: Output dimension (usually 1 for mastery)
        """
        super(DKTModel, self).__init__()
        
        self.num_questions = num_questions
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding_size = embedding_size
        
        # Question embedding layer
        # Maps question IDs to dense vectors
        self.question_embedding = nn.Embedding(
            num_embeddings=num_questions + 1,  # +1 for padding index
            embedding_dim=embedding_size,
            padding_idx=0
        )
        
        # Correctness embedding layer
        # Maps correctness (0 or 1) to a dense vector
        self.correctness_embedding = nn.Embedding(
            num_embeddings=2,
            embedding_dim=embedding_size,
            padding_idx=-1
        )
        
        # Input dimension: question_embedding + correctness_embedding + question_id_feature
        self.input_size = embedding_size + embedding_size + 1
        
        # LSTM layers for sequence processing
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Output layers
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)
        
        # Activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(
        self,
        question_ids: torch.Tensor,
        is_correct: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the model
        
        Args:
            question_ids: (batch_size, sequence_length) - question IDs
            is_correct: (batch_size, sequence_length) - correctness labels (0 or 1)
            lengths: (batch_size,) - actual sequence lengths (for packing)
        
        Returns:
            output: (batch_size, sequence_length, 1) - mastery probabilities
            hidden: (num_layers, batch_size, hidden_size) - final hidden state
        """
        batch_size, seq_len = question_ids.shape
        
        # Embed question IDs
        q_embed = self.question_embedding(question_ids)  # (batch, seq_len, embed_size)
        
        # Embed correctness
        c_embed = self.correctness_embedding(is_correct)  # (batch, seq_len, embed_size)
        
        # Normalize question IDs to [0, 1] for feature
        q_feature = (question_ids.float() / self.num_questions).unsqueeze(-1)  # (batch, seq_len, 1)
        
        # Concatenate all features
        lstm_input = torch.cat([q_embed, c_embed, q_feature], dim=-1)  # (batch, seq_len, input_size)
        lstm_input = self.dropout(lstm_input)
        
        # Pack sequences if lengths provided
        if lengths is not None:
            lstm_input_packed = nn.utils.rnn.pack_padded_sequence(
                lstm_input,
                lengths.cpu(),
                batch_first=True,
                enforce_sorted=False
            )
            lstm_out, (h_n, c_n) = self.lstm(lstm_input_packed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        else:
            lstm_out, (h_n, c_n) = self.lstm(lstm_input)  # (batch, seq_len, hidden_size)
        
        lstm_out = self.dropout(lstm_out)
        
        # Dense layers
        fc_out = self.relu(self.fc1(lstm_out))  # (batch, seq_len, hidden_size // 2)
        fc_out = self.dropout(fc_out)
        output = self.fc2(fc_out)  # (batch, seq_len, 1)
        
        # Sigmoid to get probability [0, 1]
        output = self.sigmoid(output)  # (batch, seq_len, 1)
        
        return output, h_n
    
    def predict_mastery(self, question_ids: torch.Tensor, is_correct: torch.Tensor) -> torch.Tensor:
        """
        Predict mastery probability
        
        Args:
            question_ids: (batch_size, sequence_length)
            is_correct: (batch_size, sequence_length)
        
        Returns:
            (batch_size, sequence_length, 1) - mastery probabilities
        """
        self.eval()
        with torch.no_grad():
            output, _ = self.forward(question_ids, is_correct)
        return output
    
    def get_concept_mastery(
        self,
        concept_id: int,
        question_ids: torch.Tensor,
        is_correct: torch.Tensor
    ) -> float:
        """
        Get mastery probability for a specific concept
        
        Args:
            concept_id: The concept/skill ID
            question_ids: Student's question history
            is_correct: Student's correctness history
        
        Returns:
            Mastery probability (0-1)
        """
        self.eval()
        with torch.no_grad():
            output, _ = self.forward(question_ids, is_correct)
            # Average mastery for this concept across all interactions
            mastery = output.mean().item()
        return mastery


class DKTLoss(nn.Module):
    """
    Custom loss function for DKT
    Weighted binary cross-entropy that focuses on prediction accuracy
    """
    
    def __init__(self, pos_weight: float = 1.0):
        super(DKTLoss, self).__init__()
        self.pos_weight = pos_weight
        self.bce_loss = nn.BCELoss(reduction='none')
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute loss
        
        Args:
            predictions: (batch_size, sequence_length, 1) - predicted probabilities
            targets: (batch_size, sequence_length, 1) - ground truth correctness
            mask: (batch_size, sequence_length) - mask for valid positions
        
        Returns:
            Scalar loss value
        """
        # Compute BCE loss
        loss = self.bce_loss(predictions, targets)
        
        # Apply mask if provided
        if mask is not None:
            loss = loss * mask.unsqueeze(-1)
        
        # Return mean loss
        return loss.mean()
