"""
Evaluation Script for Deep Knowledge Tracing Model
Evaluates the trained model on test dataset and computes metrics
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path
from typing import Tuple, Dict
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt

from dkt_model import DKTModel
from data_preprocessor import DataPreprocessor, DataLoader
from config import MODEL_CONFIG, TRAINING_CONFIG, DATA_CONFIG, PATHS


class DKTEvaluator:
    """
    Evaluator class for Deep Knowledge Tracing model
    """
    
    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = None
        self.preprocessor = None
        self.metrics = {}
    
    def load_model(self, filepath: str, num_questions: int):
        """
        Load trained model
        
        Args:
            filepath: Path to model checkpoint
            num_questions: Number of unique questions
        """
        print(f"Loading model from {filepath}...")
        
        self.model = DKTModel(
            num_questions=num_questions,
            hidden_size=MODEL_CONFIG['hidden_size'],
            num_layers=MODEL_CONFIG['num_layers'],
            embedding_size=50,
            dropout=MODEL_CONFIG['dropout'],
            output_size=MODEL_CONFIG['output_size']
        ).to(self.device)
        
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        print("Model loaded successfully!")
    
    def load_preprocessor(self, filepath: str):
        """Load preprocessor for data consistency"""
        print(f"Loading preprocessor from {filepath}...")
        with open(filepath, 'rb') as f:
            self.preprocessor = pickle.load(f)
        print("Preprocessor loaded successfully!")
    
    def evaluate(self, test_loader: DataLoader) -> Dict:
        """
        Evaluate model on test dataset
        
        Args:
            test_loader: DataLoader for test data
        
        Returns:
            Dictionary with evaluation metrics
        """
        print("\nEvaluating model...")
        
        self.model.eval()
        all_predictions = []
        all_targets = []
        all_probs = []
        
        with torch.no_grad():
            for batch in test_loader:
                questions = torch.tensor(batch['questions'], dtype=torch.long).to(self.device)
                correctness = torch.tensor(batch['correctness'], dtype=torch.long).to(self.device)
                targets = torch.tensor(batch['targets'], dtype=torch.float32).to(self.device)
                
                # Forward pass
                output, _ = self.model(questions, correctness)  # (batch_size, seq_len, 1)
                
                # Get predictions
                probs = output.squeeze(-1)  # (batch_size, seq_len)
                preds = (probs > 0.5).long()  # (batch_size, seq_len)
                
                all_predictions.extend(preds.cpu().numpy().flatten())
                all_targets.extend(correctness.cpu().numpy().flatten())
                all_probs.extend(probs.cpu().numpy().flatten())
        
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        all_probs = np.array(all_probs)
        
        # Compute metrics
        metrics = {
            'accuracy': accuracy_score(all_targets, all_predictions),
            'precision': precision_score(all_targets, all_predictions, zero_division=0),
            'recall': recall_score(all_targets, all_predictions, zero_division=0),
            'auc': roc_auc_score(all_targets, all_probs),
            'confusion_matrix': confusion_matrix(all_targets, all_predictions).tolist()
        }
        
        self.metrics = metrics
        return metrics
    
    def print_metrics(self):
        """Print evaluation metrics"""
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)
        
        print(f"Accuracy:  {self.metrics['accuracy']:.4f}")
        print(f"Precision: {self.metrics['precision']:.4f}")
        print(f"Recall:    {self.metrics['recall']:.4f}")
        print(f"AUC:       {self.metrics['auc']:.4f}")
        
        tn, fp, fn, tp = np.array(self.metrics['confusion_matrix']).flatten()
        print(f"\nConfusion Matrix:")
        print(f"  True Negatives:  {tn}")
        print(f"  False Positives: {fp}")
        print(f"  False Negatives: {fn}")
        print(f"  True Positives:  {tp}")
    
    def compare_datasets(
        self,
        assistments_loader: DataLoader,
        primary_loader: DataLoader
    ) -> Dict:
        """
        Compare model performance on ASSISTments vs primary dataset
        
        Args:
            assistments_loader: DataLoader for ASSISTments data
            primary_loader: DataLoader for primary dataset
        
        Returns:
            Dictionary with performance comparison
        """
        print("\nComparing performance on different datasets...")
        
        metrics_assistments = self._evaluate_on_loader(assistments_loader)
        metrics_primary = self._evaluate_on_loader(primary_loader)
        
        comparison = {
            'assistments': metrics_assistments,
            'primary': metrics_primary,
            'difference': {
                'accuracy': metrics_primary['accuracy'] - metrics_assistments['accuracy'],
                'precision': metrics_primary['precision'] - metrics_assistments['precision'],
                'recall': metrics_primary['recall'] - metrics_assistments['recall'],
                'auc': metrics_primary['auc'] - metrics_assistments['auc']
            }
        }
        
        return comparison
    
    def _evaluate_on_loader(self, loader: DataLoader) -> Dict:
        """Helper method to evaluate on a specific loader"""
        self.model.eval()
        all_predictions = []
        all_targets = []
        all_probs = []
        
        with torch.no_grad():
            for batch in loader:
                questions = torch.tensor(batch['questions'], dtype=torch.long).to(self.device)
                correctness = torch.tensor(batch['correctness'], dtype=torch.long).to(self.device)
                
                output, _ = self.model(questions, correctness)
                probs = output.squeeze(-1)
                preds = (probs > 0.5).long()
                
                all_predictions.extend(preds.cpu().numpy().flatten())
                all_targets.extend(correctness.cpu().numpy().flatten())
                all_probs.extend(probs.cpu().numpy().flatten())
        
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        all_probs = np.array(all_probs)
        
        return {
            'accuracy': accuracy_score(all_targets, all_predictions),
            'precision': precision_score(all_targets, all_predictions, zero_division=0),
            'recall': recall_score(all_targets, all_predictions, zero_division=0),
            'auc': roc_auc_score(all_targets, all_probs)
        }
    
    def save_metrics(self, filepath: str):
        """Save evaluation metrics"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.metrics, f, indent=4)
        print(f"Metrics saved to {filepath}")


def main():
    """
    Main evaluation pipeline
    
    Steps:
    1. Load trained model
    2. Load primary dataset
    3. Evaluate on primary dataset
    4. Compute metrics
    5. Save results
    """
    
    # Initialize evaluator
    print("=" * 60)
    print("DKT MODEL EVALUATION")
    print("=" * 60)
    
    evaluator = DKTEvaluator(device=TRAINING_CONFIG['device'])
    
    # Load training summary for num_questions
    with open(f"{PATHS['results_dir']}/training_summary.json", 'r') as f:
        summary = json.load(f)
    num_questions = summary['num_questions']
    
    # Load model
    print("\nSTEP 1: Loading Model")
    print("-" * 60)
    evaluator.load_model(PATHS['model_checkpoint'], num_questions)
    evaluator.load_preprocessor(f"{PATHS['results_dir']}/preprocessor.pkl")
    
    # Load primary dataset
    print("\nSTEP 2: Loading Primary Dataset")
    print("-" * 60)
    
    try:
        df_primary = evaluator.preprocessor.load_primary_data(PATHS['primary_data'])
    except FileNotFoundError:
        print(f"Warning: Primary data not found at {PATHS['primary_data']}")
        print("Skipping primary evaluation")
        return
    
    # Encode questions using the same mapping
    df_primary = evaluator.preprocessor.encode_questions(df_primary)
    
    # Create sequences
    question_seqs, correctness_seqs, student_ids = evaluator.preprocessor.create_sequences(
        df_primary,
        max_length=DATA_CONFIG['max_sequence_length'],
        min_interactions=DATA_CONFIG['min_interactions']
    )
    
    # Prepare data
    X_questions, X_correctness, lengths = evaluator.preprocessor.prepare_training_data(
        question_seqs,
        correctness_seqs,
        max_length=DATA_CONFIG['max_sequence_length']
    )
    
    targets = evaluator.preprocessor.create_targets(correctness_seqs)
    
    # Create data loader
    test_loader = DataLoader(
        X_questions, X_correctness, lengths, targets,
        batch_size=TRAINING_CONFIG['batch_size'], shuffle=False
    )
    
    # Evaluate
    print("\nSTEP 3: Evaluating Model")
    print("-" * 60)
    
    evaluator.evaluate(test_loader)
    evaluator.print_metrics()
    
    # Save results
    print("\nSTEP 4: Saving Results")
    print("-" * 60)
    
    evaluator.save_metrics(f"{PATHS['results_dir']}/evaluation_metrics.json")
    
    print("\n" + "=" * 60)
    print("Evaluation Complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
