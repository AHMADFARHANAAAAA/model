"""
Training Script for Deep Knowledge Tracing Model
Trains the DKT model on ASSISTments dataset
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import Tuple, Dict
import json
import sys

from dkt_model import DKTModel, DKTLoss
from data_preprocessor import DataPreprocessor, DataLoader
from config import MODEL_CONFIG, TRAINING_CONFIG, DATA_CONFIG, PATHS


class DKTTrainer:
    """
    Trainer class for Deep Knowledge Tracing model
    """
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
        
        print(f"Using device: {self.device}")
    
    def build_model(self, num_questions: int) -> DKTModel:
        """
        Build and initialize the DKT model
        
        Args:
            num_questions: Number of unique questions
        
        Returns:
            Initialized model
        """
        print("\nBuilding DKT Model...")
        self.model = DKTModel(
            num_questions=num_questions,
            hidden_size=MODEL_CONFIG['hidden_size'],
            num_layers=MODEL_CONFIG['num_layers'],
            embedding_size=50,
            dropout=MODEL_CONFIG['dropout'],
            output_size=MODEL_CONFIG['output_size']
        ).to(self.device)
        
        # Print model architecture
        print(self.model)
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        return self.model
    
    def setup_optimizer(self):
        """Setup optimizer and loss function"""
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=TRAINING_CONFIG['learning_rate'],
            weight_decay=TRAINING_CONFIG['weight_decay']
        )
        self.criterion = DKTLoss()
        print(f"Optimizer: Adam (lr={TRAINING_CONFIG['learning_rate']})")
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """
        Train for one epoch
        
        Args:
            train_loader: DataLoader for training data
        
        Returns:
            Tuple of (average_loss, average_accuracy)
        """
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        for batch in tqdm(train_loader, desc="Training"):
            questions = torch.tensor(batch['questions'], dtype=torch.long).to(self.device)
            correctness = torch.tensor(batch['correctness'], dtype=torch.long).to(self.device)
            targets = torch.tensor(batch['targets'], dtype=torch.float32).to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            output, _ = self.model(questions, correctness)  # (batch_size, seq_len, 1)
            
            # Compute loss
            loss = self.criterion(output, correctness.unsqueeze(-1).float())
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item() * questions.shape[0]
            
            # Accuracy: round predictions and compare
            predictions = (output > 0.5).float()
            correct = (predictions == correctness.unsqueeze(-1).float()).sum().item()
            total_correct += correct
            total_samples += questions.shape[0] * questions.shape[1]
        
        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples
        
        return avg_loss, avg_acc
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """
        Validate the model
        
        Args:
            val_loader: DataLoader for validation data
        
        Returns:
            Tuple of (average_loss, average_accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                questions = torch.tensor(batch['questions'], dtype=torch.long).to(self.device)
                correctness = torch.tensor(batch['correctness'], dtype=torch.long).to(self.device)
                targets = torch.tensor(batch['targets'], dtype=torch.float32).to(self.device)
                
                # Forward pass
                output, _ = self.model(questions, correctness)
                
                # Compute loss
                loss = self.criterion(output, correctness.unsqueeze(-1).float())
                
                # Track metrics
                total_loss += loss.item() * questions.shape[0]
                
                # Accuracy
                predictions = (output > 0.5).float()
                correct = (predictions == correctness.unsqueeze(-1).float()).sum().item()
                total_correct += correct
                total_samples += questions.shape[0] * questions.shape[1]
        
        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples
        
        return avg_loss, avg_acc
    
    def save_checkpoint(self, filepath: str):
        """Save model checkpoint"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history
        }, filepath)
        print(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint.get('history', {})
        print(f"Checkpoint loaded from {filepath}")
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        early_stopping_patience: int = 10
    ):
        """
        Train the model
        
        Args:
            train_loader: DataLoader for training
            val_loader: DataLoader for validation
            num_epochs: Number of epochs
            early_stopping_patience: Early stopping patience
        """
        best_val_loss = float('inf')
        patience_counter = 0
        
        print(f"\nStarting training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            print(f"\n--- Epoch {epoch + 1}/{num_epochs} ---")
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            
            # Validate
            val_loss, val_acc = self.validate(val_loader)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_checkpoint(PATHS['model_checkpoint'])
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"\nEarly stopping at epoch {epoch + 1}")
                    break
        
        print("\nTraining completed!")
        self.load_checkpoint(PATHS['model_checkpoint'])
    
    def save_history(self, filepath: str):
        """Save training history"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.history, f, indent=4)
        print(f"History saved to {filepath}")


def main():
    """
    Main training pipeline
    
    Steps:
    1. Load ASSISTments dataset
    2. Preprocess and create sequences
    3. Build DKT model
    4. Train the model
    5. Save results
    """
    
    # Create output directories
    Path(PATHS['model_checkpoint']).parent.mkdir(parents=True, exist_ok=True)
    Path(PATHS['results_dir']).mkdir(parents=True, exist_ok=True)
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(random_seed=DATA_CONFIG['random_seed'])
    
    # Load ASSISTments data
    print("=" * 60)
    print("STEP 1: Loading ASSISTments Dataset")
    print("=" * 60)
    
    try:
        df_assistments = preprocessor.load_assistments_data(PATHS['assistments_data'])
    except FileNotFoundError:
        print(f"Error: Dataset not found at {PATHS['assistments_data']}")
        print("Please ensure the ASSISTments CSV is in the correct location")
        sys.exit(1)
    
    # Build question mapping
    print("\n" + "=" * 60)
    print("STEP 2: Preprocessing Data")
    print("=" * 60)
    
    preprocessor.build_question_map(df_assistments)
    num_questions = len(preprocessor.question_to_id)
    print(f"Total unique questions: {num_questions}")
    
    df_assistments = preprocessor.encode_questions(df_assistments)
    
    # Create sequences
    question_seqs, correctness_seqs, student_ids = preprocessor.create_sequences(
        df_assistments,
        max_length=DATA_CONFIG['max_sequence_length'],
        min_interactions=DATA_CONFIG['min_interactions']
    )
    
    # Create targets
    targets = preprocessor.create_targets(correctness_seqs)
    
    # Prepare training data
    X_questions, X_correctness, lengths = preprocessor.prepare_training_data(
        question_seqs,
        correctness_seqs,
        max_length=DATA_CONFIG['max_sequence_length']
    )
    
    # Split into train/val/test
    (X_train_q, X_train_c), (X_val_q, X_val_c), (X_test_q, X_test_c), \
    y_train, y_val, y_test = preprocessor.split_train_val_test(
        X_questions, X_correctness, targets,
        train_ratio=0.6, val_ratio=0.2, test_ratio=0.2
    )
    
    lengths_train, lengths_val, lengths_test = lengths[:len(y_train)], \
                                                 lengths[len(y_train):len(y_train)+len(y_val)], \
                                                 lengths[len(y_train)+len(y_val):]
    
    # Create data loaders
    train_loader = DataLoader(
        X_train_q, X_train_c, lengths_train, y_train,
        batch_size=TRAINING_CONFIG['batch_size'], shuffle=True
    )
    
    val_loader = DataLoader(
        X_val_q, X_val_c, lengths_val, y_val,
        batch_size=TRAINING_CONFIG['batch_size'], shuffle=False
    )
    
    # Initialize trainer
    print("\n" + "=" * 60)
    print("STEP 3: Building Model")
    print("=" * 60)
    
    trainer = DKTTrainer(device=TRAINING_CONFIG['device'])
    trainer.build_model(num_questions)
    trainer.setup_optimizer()
    
    # Train model
    print("\n" + "=" * 60)
    print("STEP 4: Training Model")
    print("=" * 60)
    
    trainer.train(
        train_loader,
        val_loader,
        num_epochs=TRAINING_CONFIG['num_epochs'],
        early_stopping_patience=TRAINING_CONFIG['early_stopping_patience']
    )
    
    # Save results
    print("\n" + "=" * 60)
    print("STEP 5: Saving Results")
    print("=" * 60)
    
    trainer.save_history(f"{PATHS['results_dir']}/training_history.json")
    
    # Save preprocessor for later use
    import pickle
    with open(f"{PATHS['results_dir']}/preprocessor.pkl", 'wb') as f:
        pickle.dump(preprocessor, f)
    print(f"Preprocessor saved")
    
    # Save summary
    summary = {
        'num_questions': num_questions,
        'num_students': len(set(student_ids)),
        'total_interactions': len(df_assistments),
        'num_sequences': len(question_seqs),
        'model_config': MODEL_CONFIG,
        'training_config': TRAINING_CONFIG,
        'data_config': DATA_CONFIG
    }
    
    with open(f"{PATHS['results_dir']}/training_summary.json", 'w') as f:
        json.dump(summary, f, indent=4)
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Model saved to: {PATHS['model_checkpoint']}")
    print(f"Results saved to: {PATHS['results_dir']}")


if __name__ == '__main__':
    main()
