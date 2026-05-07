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
import argparse
from datetime import datetime
from sklearn.metrics import roc_auc_score, precision_score, recall_score, precision_recall_curve, roc_curve

from dkt_model import DKTModel, DKTLoss
from data_preprocessor import DataPreprocessor, DataLoader
from config import MODEL_CONFIG, TRAINING_CONFIG, DATA_CONFIG, PATHS


class DKTTrainer:
    """
    Trainer class for Deep Knowledge Tracing model
    """
    
    def __init__(self, device: str = 'auto'):
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        elif device == 'cuda':
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA selected but not available. "
                    "Use --device cpu or install CUDA-enabled PyTorch."
                )
            self.device = 'cuda'
        elif device == 'cpu':
            self.device = 'cpu'
        else:
            raise ValueError("device must be one of: auto, cuda, cpu")
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.freeze_encoder_epochs = 0
        self.encoder_trainable = True
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'train_auc': [],
            'val_auc': [],
            'train_precision': [],
            'val_precision': [],
            'train_recall': [],
            'val_recall': [],
            'decision_threshold': []
        }
        self.decision_threshold = float(TRAINING_CONFIG.get('decision_threshold', 0.5))
        # AMP / scaler
        self.use_amp = False
        self.scaler = None
        if self.device == 'cuda' and TRAINING_CONFIG.get('use_amp', False):
            self.use_amp = True
            try:
                self.scaler = torch.amp.GradScaler('cuda', enabled=True)
            except Exception:
                self.scaler = None
                self.use_amp = False
        
        print(f"Using device: {self.device}")

    def set_finetune_schedule(self, freeze_encoder_epochs: int = 0):
        """Configure staged fine-tuning for transfer learning runs."""
        self.freeze_encoder_epochs = max(0, int(freeze_encoder_epochs))
        self.encoder_trainable = self.freeze_encoder_epochs == 0

    def _set_encoder_trainable(self, trainable: bool):
        """Enable or disable gradients for the sequence encoder stack."""
        self.encoder_trainable = bool(trainable)
        if self.model is None:
            return

        encoder_modules = [
            self.model.question_embedding,
            self.model.correctness_embedding,
            self.model.lstm,
        ]
        for module in encoder_modules:
            for parameter in module.parameters():
                parameter.requires_grad = trainable

    def _get_trainable_parameter_groups(self):
        """Split encoder and prediction head parameters for fine-tuning."""
        encoder_params = []
        head_params = []

        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            if name.startswith('fc1') or name.startswith('fc2'):
                head_params.append(parameter)
            else:
                encoder_params.append(parameter)

        return encoder_params, head_params
    
    def build_model(self, num_questions: int, num_topics: int | None = None) -> DKTModel:
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
            num_topics=num_topics,
            hidden_size=MODEL_CONFIG['hidden_size'],
            num_layers=MODEL_CONFIG['num_layers'],
            embedding_size=MODEL_CONFIG.get('embedding_size', 50),
            dropout=MODEL_CONFIG['dropout'],
            output_size=MODEL_CONFIG['output_size'],
            encoder_type=MODEL_CONFIG.get('encoder_type', 'lstm'),
            use_attention=MODEL_CONFIG.get('use_attention', False),
            interaction_feature_dim=MODEL_CONFIG.get('interaction_feature_dim', 0),
            use_layer_norm=MODEL_CONFIG.get('use_layer_norm', True),
            residual_ffn_multiplier=MODEL_CONFIG.get('residual_ffn_multiplier', 2),
            head_dropout=MODEL_CONFIG.get('head_dropout', 0.25),
        ).to(self.device)
        
        # Print model architecture
        print(self.model)
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        return self.model
    
    def setup_optimizer(
        self,
        pos_weight: float = 1.0,
        encoder_lr_multiplier: float = 1.0,
        head_lr_multiplier: float = 1.0,
    ):
        """Setup optimizer and loss function"""
        base_lr = float(TRAINING_CONFIG['learning_rate'])
        encoder_lr = base_lr * float(encoder_lr_multiplier)
        head_lr = base_lr * float(head_lr_multiplier)

        encoder_params, head_params = self._get_trainable_parameter_groups()
        param_groups = []
        if encoder_params:
            param_groups.append({'params': encoder_params, 'lr': encoder_lr})
        if head_params:
            param_groups.append({'params': head_params, 'lr': head_lr})
        if not param_groups:
            param_groups = [{'params': self.model.parameters(), 'lr': base_lr}]

        self.optimizer = optim.Adam(
            param_groups,
            weight_decay=TRAINING_CONFIG['weight_decay']
        )
        self.criterion = DKTLoss(
            pos_weight=pos_weight,
            label_smoothing=float(TRAINING_CONFIG.get('label_smoothing', 0.0)),
            loss_type=str(TRAINING_CONFIG.get('loss_type', 'bce')),
            focal_gamma=float(TRAINING_CONFIG.get('focal_gamma', 2.0)),
            focal_alpha=float(TRAINING_CONFIG.get('focal_alpha', 0.25)),
        )
        if TRAINING_CONFIG.get('use_reduce_lr_on_plateau', True):
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=float(TRAINING_CONFIG.get('scheduler_factor', 0.5)),
                patience=int(TRAINING_CONFIG.get('scheduler_patience', 1)),
                min_lr=float(TRAINING_CONFIG.get('scheduler_min_lr', 1e-5))
            )
        print(f"Optimizer: Adam (encoder_lr={encoder_lr}, head_lr={head_lr})")
        print(f"Loss pos_weight: {pos_weight:.4f}")
        print(f"Label smoothing: {float(TRAINING_CONFIG.get('label_smoothing', 0.0)):.3f}")
        print(f"Loss type: {str(TRAINING_CONFIG.get('loss_type', 'bce')).lower()}")

    @staticmethod
    def _build_sequence_binary_labels(correctness: np.ndarray, lengths: np.ndarray) -> np.ndarray:
        """Create sequence-level binary labels for sampler balancing."""
        labels = []
        for i in range(len(lengths)):
            seq_len = int(lengths[i])
            if seq_len < 2:
                labels.append(0)
                continue
            # Use next-step targets only to avoid padded region.
            seq_targets = correctness[i, 1:seq_len]
            labels.append(int(np.mean(seq_targets) >= 0.5))
        return np.asarray(labels, dtype=np.int64)
    
    @staticmethod
    def _compute_auc(targets: np.ndarray, probs: np.ndarray) -> float:
        """Compute ROC AUC safely when both classes are present."""
        unique_targets = np.unique(targets)
        if unique_targets.shape[0] < 2:
            return float('nan')
        return float(roc_auc_score(targets, probs))

    @staticmethod
    def _compute_precision_recall(targets: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> Tuple[float, float]:
        """Compute precision and recall from probability outputs."""
        predictions = (probs > threshold).astype(int)
        targets = targets.astype(int)
        precision = float(precision_score(targets, predictions, zero_division=0))
        recall = float(recall_score(targets, predictions, zero_division=0))
        return precision, recall

    @staticmethod
    def _find_optimal_threshold(
        targets: np.ndarray,
        probs: np.ndarray,
        strategy: str = 'pr',
        min_precision: float = 0.60,
        min_recall: float = 0.20,
        beta: float = 1.0,
        default_threshold: float = 0.5,
    ) -> float:
        """Pick threshold from PR/ROC curve while respecting precision/recall floors."""
        if len(np.unique(targets)) < 2:
            return default_threshold

        strategy = str(strategy).lower()
        targets_int = targets.astype(int)

        def evaluate_threshold(threshold_value: float) -> Tuple[float, float, float]:
            preds = (probs >= threshold_value).astype(int)

            tp = float(np.sum((preds == 1) & (targets_int == 1)))
            fp = float(np.sum((preds == 1) & (targets_int == 0)))
            tn = float(np.sum((preds == 0) & (targets_int == 0)))
            fn = float(np.sum((preds == 0) & (targets_int == 1)))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            return precision, recall, specificity

        if strategy == 'roc':
            fpr, tpr, thresholds = roc_curve(targets_int, probs)
            if thresholds.size == 0:
                return default_threshold

            best_threshold = default_threshold
            best_score = -1.0
            for threshold, fpr_i, tpr_i in zip(thresholds, fpr, tpr):
                precision, recall, _ = evaluate_threshold(float(threshold))
                if precision < min_precision or recall < min_recall:
                    continue

                # Maximize Youden's J with floor constraints.
                score = float(tpr_i - fpr_i)
                if score > best_score:
                    best_score = score
                    best_threshold = float(np.clip(threshold, 0.0, 1.0))

            if best_score >= 0:
                return best_threshold

            return default_threshold

        _, _, thresholds = precision_recall_curve(targets.astype(int), probs)
        if thresholds.size == 0:
            return default_threshold

        thresholds = np.unique(np.clip(thresholds, 0.0, 1.0))

        best_threshold = default_threshold
        best_score = -1.0
        beta = max(1e-6, float(beta))
        beta_sq = beta * beta

        for threshold in thresholds:
            precision, recall, specificity = evaluate_threshold(float(threshold))
            if precision < min_precision or recall < min_recall:
                continue

            # Optimize F-beta then tie-break by balanced accuracy.
            fbeta_denom = (beta_sq * precision + recall)
            fbeta = ((1 + beta_sq) * precision * recall / fbeta_denom) if fbeta_denom > 0 else 0.0
            balanced_acc = 0.5 * (recall + specificity)
            score = fbeta + 0.05 * balanced_acc

            if score > best_score:
                best_score = score
                best_threshold = float(threshold)

        if best_score >= 0:
            return best_threshold

        # Fallback if floor constraints are too strict.
        for threshold in thresholds:
            precision, recall, specificity = evaluate_threshold(float(threshold))
            balanced_acc = 0.5 * (recall + specificity)
            fbeta_denom = (beta_sq * precision + recall)
            fbeta = ((1 + beta_sq) * precision * recall / fbeta_denom) if fbeta_denom > 0 else 0.0
            score = fbeta + 0.05 * balanced_acc
            if score > best_score:
                best_score = score
                best_threshold = float(threshold)

        return best_threshold

    @staticmethod
    def _estimate_pos_weight(correctness: np.ndarray, lengths: np.ndarray, max_pos_weight: float = 5.0) -> float:
        """Estimate positive class weight from next-step train targets."""
        positives = 0
        negatives = 0

        for i in range(len(lengths)):
            seq_len = int(lengths[i])
            if seq_len < 2:
                continue

            targets = correctness[i, 1:seq_len]
            positives += int(np.sum(targets == 1))
            negatives += int(np.sum(targets == 0))

        if positives == 0 or negatives == 0:
            return 1.0

        raw_weight = negatives / positives
        return float(np.clip(raw_weight, 1.0, max_pos_weight))

    @staticmethod
    def _build_next_step_batch(
        questions: torch.Tensor,
        correctness: torch.Tensor,
        topics: torch.Tensor,
        interaction_features: torch.Tensor,
        lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build next-step DKT inputs, targets, and mask."""
        next_questions = questions[:, :-1]
        next_topics = topics[:, :-1]
        next_interaction_features = interaction_features[:, :-1]
        next_targets = correctness[:, 1:].float().unsqueeze(-1)

        max_valid_len = next_targets.shape[1]
        valid_lengths = torch.clamp(lengths - 1, min=0, max=max_valid_len)
        positions = torch.arange(max_valid_len, device=questions.device).unsqueeze(0)
        mask = positions < valid_lengths.unsqueeze(1)

        return next_questions, next_topics, next_interaction_features, next_targets, mask

    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float, float, float, float]:
        """
        Train for one epoch
        
        Args:
            train_loader: DataLoader for training data
        
        Returns:
            Tuple of (average_loss, average_accuracy, average_auc, average_precision, average_recall)
        """
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        all_probs = []
        all_targets = []
        
        for batch in tqdm(train_loader, desc="  Training", ncols=100, position=0, leave=True):
            questions = batch['questions']
            if not isinstance(questions, torch.Tensor):
                questions = torch.tensor(questions, dtype=torch.long)
            questions = questions.to(self.device, non_blocking=True)

            correctness = batch['correctness']
            if not isinstance(correctness, torch.Tensor):
                correctness = torch.tensor(correctness, dtype=torch.long)
            correctness = correctness.to(self.device, non_blocking=True)

            topics = batch.get('topics')
            if topics is None:
                topics = torch.zeros_like(questions)
            if not isinstance(topics, torch.Tensor):
                topics = torch.tensor(topics, dtype=torch.long)
            topics = topics.to(self.device, non_blocking=True)

            interaction_features = batch.get('interaction_features')
            if interaction_features is None:
                interaction_features = torch.zeros(questions.size(0), questions.size(1), int(MODEL_CONFIG.get('interaction_feature_dim', 0)), dtype=torch.float32)
            if not isinstance(interaction_features, torch.Tensor):
                interaction_features = torch.tensor(interaction_features, dtype=torch.float32)
            interaction_features = interaction_features.to(self.device, non_blocking=True)

            lengths = batch['lengths']
            if not isinstance(lengths, torch.Tensor):
                lengths = torch.tensor(lengths, dtype=torch.long)
            lengths = lengths.to(self.device, non_blocking=True)

            if torch.any(lengths < 2):
                continue

            input_questions, input_topics, input_interaction_features, next_targets, mask = self._build_next_step_batch(questions, correctness, topics, interaction_features, lengths)
            
            # Forward / backward with optional AMP
            self.optimizer.zero_grad()
            if self.use_amp:
                with torch.amp.autocast('cuda', enabled=True):
                    output, _ = self.model(input_questions, correctness[:, :-1], topic_ids=input_topics, interaction_features=input_interaction_features, lengths=lengths - 1)
                    probs = torch.sigmoid(output)
                    loss = self.criterion(output, next_targets, mask=mask)

                # Scaled backward
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
            else:
                output, _ = self.model(input_questions, correctness[:, :-1], topic_ids=input_topics, interaction_features=input_interaction_features, lengths=lengths - 1)
                probs = torch.sigmoid(output)
                loss = self.criterion(output, next_targets, mask=mask)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item() * mask.sum().item()
            
            # Accuracy: round predictions and compare
            predictions = (probs > self.decision_threshold).float()
            correct = ((predictions == next_targets).float() * mask.unsqueeze(-1).float()).sum().item()
            total_correct += correct
            total_samples += mask.sum().item()

            masked_probs = probs.detach().squeeze(-1)[mask].cpu().numpy().flatten()
            masked_targets = next_targets.detach().squeeze(-1)[mask].cpu().numpy().flatten()
            all_probs.extend(masked_probs)
            all_targets.extend(masked_targets)
        
        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples
        targets_array = np.array(all_targets)
        probs_array = np.array(all_probs)
        avg_auc = self._compute_auc(targets_array, probs_array)
        avg_precision, avg_recall = self._compute_precision_recall(
            targets_array,
            probs_array,
            threshold=self.decision_threshold,
        )
        
        return avg_loss, avg_acc, avg_auc, avg_precision, avg_recall
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, float, float, float, float, float]:
        """
        Validate the model
        
        Args:
            val_loader: DataLoader for validation data
        
        Returns:
            Tuple of (average_loss, average_accuracy, average_auc, average_precision, average_recall, tuned_threshold)
        """
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        all_probs = []
        all_targets = []
        
        # Validation - use autocast for faster FP16 inference if enabled
        for batch in tqdm(val_loader, desc="  Validating", ncols=100, position=0, leave=True):
            # Handle tensors or numpy arrays (support both custom loader and torch DataLoader)
            if not isinstance(batch['questions'], torch.Tensor):
                questions = torch.tensor(batch['questions'], dtype=torch.long)
            else:
                questions = batch['questions']
            questions = questions.to(self.device, non_blocking=True)

            if not isinstance(batch['correctness'], torch.Tensor):
                correctness = torch.tensor(batch['correctness'], dtype=torch.long)
            else:
                correctness = batch['correctness']
            correctness = correctness.to(self.device, non_blocking=True)

            topics = batch.get('topics')
            if topics is None:
                topics = torch.zeros_like(questions)
            if not isinstance(topics, torch.Tensor):
                topics = torch.tensor(topics, dtype=torch.long)
            topics = topics.to(self.device, non_blocking=True)

            interaction_features = batch.get('interaction_features')
            if interaction_features is None:
                interaction_features = torch.zeros(questions.size(0), questions.size(1), int(MODEL_CONFIG.get('interaction_feature_dim', 0)), dtype=torch.float32)
            if not isinstance(interaction_features, torch.Tensor):
                interaction_features = torch.tensor(interaction_features, dtype=torch.float32)
            interaction_features = interaction_features.to(self.device, non_blocking=True)

            if not isinstance(batch['lengths'], torch.Tensor):
                lengths = torch.tensor(batch['lengths'], dtype=torch.long)
            else:
                lengths = batch['lengths']
            lengths = lengths.to(self.device, non_blocking=True)

            if torch.any(lengths < 2):
                continue

            input_questions, input_topics, input_interaction_features, next_targets, mask = self._build_next_step_batch(questions, correctness, topics, interaction_features, lengths)

            with torch.no_grad():
                if self.use_amp:
                    with torch.amp.autocast('cuda', enabled=True):
                        output, _ = self.model(input_questions, correctness[:, :-1], topic_ids=input_topics, interaction_features=input_interaction_features, lengths=lengths - 1)
                        probs = torch.sigmoid(output)
                        loss = self.criterion(output, next_targets, mask=mask)
                else:
                    output, _ = self.model(input_questions, correctness[:, :-1], topic_ids=input_topics, interaction_features=input_interaction_features, lengths=lengths - 1)
                    probs = torch.sigmoid(output)
                    loss = self.criterion(output, next_targets, mask=mask)

            # Track metrics
            total_loss += loss.item() * mask.sum().item()

            # Accuracy
            predictions = (probs > self.decision_threshold).float()
            correct = ((predictions == next_targets).float() * mask.unsqueeze(-1).float()).sum().item()
            total_correct += correct
            total_samples += mask.sum().item()

            masked_probs = probs.detach().squeeze(-1)[mask].cpu().numpy().flatten()
            masked_targets = next_targets.detach().squeeze(-1)[mask].cpu().numpy().flatten()
            all_probs.extend(masked_probs)
            all_targets.extend(masked_targets)
        
        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples
        targets_array = np.array(all_targets)
        probs_array = np.array(all_probs)
        avg_auc = self._compute_auc(targets_array, probs_array)
        min_precision = float(TRAINING_CONFIG.get('min_precision_for_threshold_tuning', 0.60))
        min_recall = float(TRAINING_CONFIG.get('min_recall_for_threshold_tuning', 0.20))
        threshold_strategy = str(TRAINING_CONFIG.get('threshold_strategy', 'pr')).lower()
        threshold_beta = float(TRAINING_CONFIG.get('threshold_beta', 1.0))
        tuned_threshold = self._find_optimal_threshold(
            targets_array,
            probs_array,
            strategy=threshold_strategy,
            min_precision=min_precision,
            min_recall=min_recall,
            beta=threshold_beta,
            default_threshold=self.decision_threshold,
        )
        avg_precision, avg_recall = self._compute_precision_recall(
            targets_array,
            probs_array,
            threshold=tuned_threshold,
        )
        
        return avg_loss, avg_acc, avg_auc, avg_precision, avg_recall, tuned_threshold
    
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
        """Load model checkpoint and accumulated history"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        if self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load history from checkpoint
        checkpoint_history = checkpoint.get('history', {})
        
        # Merge with current history (accumulate)
        if checkpoint_history:
            for key in checkpoint_history:
                if key not in self.history:
                    self.history[key] = []
                # Extend only if not already present (avoid duplicates)
                if not self.history[key]:
                    self.history[key] = checkpoint_history[key]
        
        print(f"Checkpoint loaded from {filepath}")

    def load_pretrained_weights(self, filepath: str):
        """Load only compatible model weights for transfer learning."""
        checkpoint = torch.load(filepath, map_location=self.device)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        model_state = self.model.state_dict()

        loaded_keys = []
        skipped_keys = []

        for key, value in state_dict.items():
            if key in model_state and model_state[key].shape == value.shape:
                model_state[key] = value
                loaded_keys.append(key)
            else:
                skipped_keys.append(key)

        self.model.load_state_dict(model_state)
        print(f"Loaded pretrained weights from {filepath}")
        print(f"  Loaded tensors: {len(loaded_keys)}")
        if skipped_keys:
            print(f"  Skipped tensors: {len(skipped_keys)}")
            print("  Note: question_embedding.weight is usually skipped when dataset vocabularies differ.")
    
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
        best_val_auc_ckpt = float('-inf')
        patience_counter = 0
        # Overfitting detector: if validation loss increases while training loss decreases
        overfit_counter = 0
        overfit_patience = TRAINING_CONFIG.get('overfitting_patience', 3)
        overfit_delta = TRAINING_CONFIG.get('overfit_delta', 0.001)
        prev_train_loss = None
        prev_val_loss = None
        
        # No improvement detector: if metrics don't improve
        no_improvement_counter = 0
        no_improvement_patience = TRAINING_CONFIG.get('no_improvement_patience', 5)
        min_performance_threshold = TRAINING_CONFIG.get('min_performance_threshold', 0.5)
        best_val_acc = 0.0
        best_val_auc = 0.0
        
        print(f"\n{'='*80}")
        print(f"Starting training for {num_epochs} epochs...")
        print(f"{'='*80}")
        checkpoint_metric = str(TRAINING_CONFIG.get('checkpoint_metric', 'val_auc')).lower()

        if self.freeze_encoder_epochs > 0:
            self._set_encoder_trainable(False)
            print(f"Transfer learning: encoder frozen for first {self.freeze_encoder_epochs} epoch(s)")
        
        for epoch in range(num_epochs):
            print(f"\n{'='*80}")
            print(f"🔄 Epoch {epoch + 1}/{num_epochs}")
            print(f"{'='*80}")

            if self.freeze_encoder_epochs > 0 and epoch == self.freeze_encoder_epochs:
                self._set_encoder_trainable(True)
                print("Transfer learning: encoder unfrozen for joint fine-tuning")
            
            # Train
            train_loss, train_acc, train_auc, train_precision, train_recall = self.train_epoch(train_loader)
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['train_auc'].append(train_auc)
            self.history['train_precision'].append(train_precision)
            self.history['train_recall'].append(train_recall)
            
            # Validate
            val_loss, val_acc, val_auc, val_precision, val_recall, tuned_threshold = self.validate(val_loader)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['val_auc'].append(val_auc)
            self.history['val_precision'].append(val_precision)
            self.history['val_recall'].append(val_recall)
            self.decision_threshold = tuned_threshold
            self.history['decision_threshold'].append(tuned_threshold)
            
            print(f"\n📊 TRAIN METRICS:")
            print(f"   Loss: {train_loss:.4f} | Accuracy: {train_acc:.4f} | AUC: {train_auc:.4f}")
            print(f"   Precision: {train_precision:.4f} | Recall: {train_recall:.4f}")
            print(f"\n📈 VALIDATION METRICS:")
            print(f"   Loss: {val_loss:.4f} | Accuracy: {val_acc:.4f} | AUC: {val_auc:.4f}")
            print(f"   Precision: {val_precision:.4f} | Recall: {val_recall:.4f}")
            print(f"   Tuned threshold: {tuned_threshold:.4f}")

            if self.scheduler is not None:
                self.scheduler.step(val_loss)
                current_lr = float(self.optimizer.param_groups[0]['lr'])
                print(f"   Learning rate: {current_lr:.6f}")
            
            # Display gap indicator
            gap = val_loss - train_loss
            gap_status = "⚠️ HIGH" if gap > 0.05 else "⚠️ MODERATE" if gap > 0.01 else "✓ GOOD"
            print(f"\n📏 LOSS GAP (Val - Train): {gap:+.4f} {gap_status}")
            
            # Early stopping and checkpointing
            improved_for_early_stop = val_loss < best_val_loss
            if improved_for_early_stop:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                print(f"\n❌ Validation loss not improved (patience: {patience_counter}/{early_stopping_patience})")
                if patience_counter >= early_stopping_patience:
                    print(f"\n{'⏹️ '*40}")
                    print(f"🛑 EARLY STOPPING TRIGGERED at Epoch {epoch + 1}")
                    print(f"Reason: Validation loss not improved for {early_stopping_patience} consecutive epochs")
                    print(f"{'⏹️ '*40}\n")
                    break

            if checkpoint_metric == 'val_loss':
                improved_for_ckpt = improved_for_early_stop
            else:
                improved_for_ckpt = val_auc > best_val_auc_ckpt

            if improved_for_ckpt:
                if checkpoint_metric != 'val_loss':
                    best_val_auc_ckpt = val_auc
                self.save_checkpoint(PATHS['model_checkpoint'])
                print(f"\n✅ Checkpoint updated by {checkpoint_metric}: "
                      f"{val_auc:.4f}" if checkpoint_metric == 'val_auc' else
                      f"\n✅ Checkpoint updated by {checkpoint_metric}: {val_loss:.4f}")

            # Overfitting detection: val loss increases while train loss decreases
            if prev_train_loss is not None and prev_val_loss is not None:
                if (val_loss > prev_val_loss) and (train_loss < prev_train_loss) and ((val_loss - train_loss) > overfit_delta):
                    overfit_counter += 1
                    print(f"⚠️  OVERFITTING WARNING: Epoch {epoch+1} (severity {overfit_counter}/{overfit_patience})")
                else:
                    overfit_counter = 0

                if overfit_counter >= overfit_patience:
                    print(f"\n{'🚨 '*40}")
                    print(f"🚨 OVERFITTING DETECTED at Epoch {epoch + 1}")
                    print(f"Reason: Validation loss increased while training loss decreased for {overfit_patience} consecutive epochs")
                    print(f"Training will stop to prevent further overfitting")
                    print(f"{'🚨 '*40}\n")
                    break
            # Poor performance detection: stop if val_acc or val_auc below minimum
            if val_acc < min_performance_threshold or val_auc < min_performance_threshold:
                print(f"\n{'❌ '*40}")
                print(f"❌ POOR PERFORMANCE DETECTED at Epoch {epoch + 1}")
                print(f"Reason: Validation accuracy ({val_acc:.4f}) or AUC ({val_auc:.4f}) below threshold ({min_performance_threshold})")
                print(f"Training will stop - model is not learning effectively")
                print(f"{'❌ '*40}\n")
                break

            # No improvement detection: stop if metrics haven't improved
            if val_acc > best_val_acc or val_auc > best_val_auc:
                # At least one metric improved
                best_val_acc = max(best_val_acc, val_acc)
                best_val_auc = max(best_val_auc, val_auc)
                no_improvement_counter = 0
                print(f"\n🎯 Metrics improved! (Acc: {val_acc:.4f} | AUC: {val_auc:.4f})")
            else:
                # No metric improvement
                no_improvement_counter += 1
                print(f"\n⚠️  NO IMPROVEMENT: Epoch {epoch + 1} (counter {no_improvement_counter}/{no_improvement_patience})")
                if no_improvement_counter >= no_improvement_patience:
                    print(f"\n{'⏸️ '*40}")
                    print(f"⏸️  STOPPING - NO IMPROVEMENT for {no_improvement_patience} consecutive epochs")
                    print(f"Best validation metrics - Acc: {best_val_acc:.4f} | AUC: {best_val_auc:.4f}")
                    print(f"{'⏸️ '*40}\n")
                    break
            prev_train_loss = train_loss
            prev_val_loss = val_loss
        
        print(f"\n{'='*80}")
        print(f"✓ Training completed! Loading best model from checkpoint...")
        print(f"{'='*80}")
        self.load_checkpoint(PATHS['model_checkpoint'])
    
    def save_history(self, filepath: str):
        """Save training history (accumulated without overwriting)"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing history if it exists
        all_histories = []
        if Path(filepath).exists():
            try:
                with open(filepath, 'r') as f:
                    existing_data = json.load(f)
                    # Check if it's a list of runs or a single run
                    if isinstance(existing_data, list):
                        all_histories = existing_data
                    else:
                        # Old format - wrap it
                        all_histories = [{
                            'run_timestamp': 'N/A (old format)',
                            'history': existing_data
                        }]
            except json.JSONDecodeError:
                print(f"Warning: Could not read existing history file, starting fresh")
                all_histories = []
        
        # Add current training run with timestamp
        current_run = {
            'run_timestamp': datetime.now().isoformat(timespec='seconds'),
            'history': self.history
        }
        all_histories.append(current_run)
        
        # Save accumulated histories
        with open(filepath, 'w') as f:
            json.dump(all_histories, f, indent=4)
        print(f"History saved to {filepath} (total runs: {len(all_histories)})")
    
    @staticmethod
    def get_accumulated_history_stats(filepath: str) -> Dict:
        """Get statistics from accumulated history file"""
        if not Path(filepath).exists():
            return {}
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    total_epochs = 0
                    run_timestamps = []
                    
                    for run in data:
                        if isinstance(run, dict) and 'history' in run:
                            # New format with 'history' key
                            run_timestamps.append(run.get('run_timestamp', 'N/A'))
                            total_epochs += len(run['history'].get('train_loss', []))
                        elif isinstance(run, dict) and 'train_loss' in run:
                            # Old format without 'history' wrapper
                            run_timestamps.append('N/A (old format)')
                            total_epochs += len(run.get('train_loss', []))
                    
                    return {
                        'total_runs': len(data),
                        'run_timestamps': run_timestamps,
                        'total_epochs': total_epochs
                    }
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            # Silently handle parse errors - don't print warnings
            pass
        
        return {}


def main():
    """
    Main training pipeline
    
    Steps:
    1. Load primary student dataset
    2. Preprocess and create sequences
    3. Build DKT model
    4. Train the model
    5. Save results
    """
    
    parser = argparse.ArgumentParser(description='Train DKT model')
    parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default=TRAINING_CONFIG['device'])
    parser.add_argument('--epochs', type=int, default=TRAINING_CONFIG['num_epochs'])
    parser.add_argument('--data-source', choices=['assistments', 'primary'], default=TRAINING_CONFIG['data_source'])
    parser.add_argument('--init-checkpoint', type=str, default=None, help='Optional pretrained checkpoint to fine-tune from')
    parser.add_argument('--checkpoint-path', type=str, default=PATHS['model_checkpoint'], help='Path where the best checkpoint is saved')
    parser.add_argument('--learning-rate', type=float, default=None, help='Override the configured learning rate')
    parser.add_argument('--robust-preset', action='store_true', help='Apply robust defaults (balanced threshold tuning, no weighted sampler)')
    parser.add_argument('--threshold-strategy', choices=['pr', 'roc'], default=None, help='Override threshold tuning strategy')
    parser.add_argument('--threshold-beta', type=float, default=None, help='Override F-beta for PR threshold tuning')
    parser.add_argument('--min-precision', type=float, default=None, help='Override minimum precision constraint for threshold tuning')
    parser.add_argument('--min-recall', type=float, default=None, help='Override minimum recall constraint for threshold tuning')
    parser.add_argument('--use-weighted-sampler', choices=['auto', 'true', 'false'], default='auto', help='Override weighted sampler usage')
    parser.add_argument('--disable-layer-norm', action='store_true', help='Disable layer norm for a simpler/more stable head during ablation')
    parser.add_argument('--freeze-encoder-epochs', type=int, default=0, help='Freeze encoder layers for the first N epochs during fine-tuning')
    parser.add_argument('--encoder-lr-multiplier', type=float, default=0.2, help='Learning-rate multiplier for encoder layers during fine-tuning')
    parser.add_argument('--head-lr-multiplier', type=float, default=1.0, help='Learning-rate multiplier for prediction head during fine-tuning')
    args = parser.parse_args()

    if args.robust_preset:
        TRAINING_CONFIG['threshold_strategy'] = 'pr'
        TRAINING_CONFIG['threshold_beta'] = 1.2
        TRAINING_CONFIG['min_precision_for_threshold_tuning'] = 0.68
        TRAINING_CONFIG['min_recall_for_threshold_tuning'] = 0.45
        TRAINING_CONFIG['use_weighted_sampler'] = False

    if args.learning_rate is not None:
        TRAINING_CONFIG['learning_rate'] = float(args.learning_rate)
    if args.threshold_strategy is not None:
        TRAINING_CONFIG['threshold_strategy'] = str(args.threshold_strategy)
    if args.threshold_beta is not None:
        TRAINING_CONFIG['threshold_beta'] = float(args.threshold_beta)
    if args.min_precision is not None:
        TRAINING_CONFIG['min_precision_for_threshold_tuning'] = float(args.min_precision)
    if args.min_recall is not None:
        TRAINING_CONFIG['min_recall_for_threshold_tuning'] = float(args.min_recall)
    if args.use_weighted_sampler != 'auto':
        TRAINING_CONFIG['use_weighted_sampler'] = args.use_weighted_sampler == 'true'
    if args.disable_layer_norm:
        MODEL_CONFIG['use_layer_norm'] = False
    TRAINING_CONFIG['data_source'] = args.data_source
    PATHS['model_checkpoint'] = args.checkpoint_path

    print("Runtime robust knobs:")
    print(
        f"  threshold_strategy={TRAINING_CONFIG.get('threshold_strategy')} "
        f"beta={TRAINING_CONFIG.get('threshold_beta')} "
        f"min_precision={TRAINING_CONFIG.get('min_precision_for_threshold_tuning')} "
        f"min_recall={TRAINING_CONFIG.get('min_recall_for_threshold_tuning')} "
        f"weighted_sampler={TRAINING_CONFIG.get('use_weighted_sampler')} "
        f"use_layer_norm={MODEL_CONFIG.get('use_layer_norm')}"
    )

    # Create output directories
    Path(PATHS['model_checkpoint']).parent.mkdir(parents=True, exist_ok=True)
    Path(PATHS['results_dir']).mkdir(parents=True, exist_ok=True)
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(random_seed=DATA_CONFIG['random_seed'])
    
    # Load primary student data
    print("=" * 60)
    print("STEP 1: Loading Student Dataset")
    print("=" * 60)
    
    try:
        data_source = str(TRAINING_CONFIG.get('data_source', 'assistments')).lower()
        if data_source == 'primary':
            dataset_path = PATHS['primary_data']
            print(f"Selected dataset source: primary ({dataset_path})")
            df_assistments = preprocessor.load_primary_data(dataset_path)
        else:
            dataset_path = PATHS['assistments_data']
            print(f"Selected dataset source: assistments ({dataset_path})")
            df_assistments = preprocessor.load_assistments_data(dataset_path)
    except FileNotFoundError:
        print(f"Error: Dataset not found at {dataset_path}")
        print("Please ensure the dataset CSV is in the correct location")
        sys.exit(1)
    
    # Build question mapping
    print("\n" + "=" * 60)
    print("STEP 2: Preprocessing Data")
    print("=" * 60)
    
    preprocessor.build_question_map(df_assistments)
    preprocessor.build_topic_map(df_assistments)
    num_questions = len(preprocessor.question_to_id)
    num_topics = len(preprocessor.topic_to_id) if preprocessor.topic_to_id else 1
    print(f"Total unique questions: {num_questions}")
    print(f"Total unique topics: {num_topics}")
    
    df_assistments = preprocessor.encode_questions(df_assistments)
    df_assistments = preprocessor.encode_topics(df_assistments)
    
    # Create sequences
    question_seqs, correctness_seqs, topic_seqs, interaction_seqs, student_ids = preprocessor.create_sequences(
        df_assistments,
        max_length=DATA_CONFIG['max_sequence_length'],
        min_interactions=DATA_CONFIG['min_interactions'],
        stride=DATA_CONFIG.get('window_stride', DATA_CONFIG['max_sequence_length'])
    )
    
    # Create targets
    targets = preprocessor.create_targets(correctness_seqs)
    
    # Prepare training data
    X_questions, X_correctness, X_topics, X_interactions, lengths = preprocessor.prepare_training_data(
        question_seqs,
        correctness_seqs,
        topic_seqs,
        interaction_seqs,
        max_length=DATA_CONFIG['max_sequence_length']
    )
    
    # Split into train/val/test
    (X_train_q, X_train_c), (X_val_q, X_val_c), (X_test_q, X_test_c), \
    X_train_t, X_val_t, X_test_t, X_train_i, X_val_i, X_test_i, y_train, y_val, y_test, lengths_train, lengths_val, lengths_test = preprocessor.split_train_val_test(
        X_questions, X_correctness, X_topics, X_interactions, lengths, targets, student_ids,
        train_ratio=0.6, val_ratio=0.2, test_ratio=0.2
    )
    
    # Create data loaders
    # Create data loaders. Prefer torch.utils.data.DataLoader when configured.
    use_torch_loader = TRAINING_CONFIG.get('use_torch_dataloader', True)
    if use_torch_loader:
        try:
            from data_preprocessor import TorchDataset
            train_dataset = TorchDataset(X_train_q, X_train_c, X_train_t, X_train_i, lengths_train, y_train)
            val_dataset = TorchDataset(X_val_q, X_val_c, X_val_t, X_val_i, lengths_val, y_val)
            use_weighted_sampler = bool(TRAINING_CONFIG.get('use_weighted_sampler', False))
            if use_weighted_sampler:
                seq_labels = DKTTrainer._build_sequence_binary_labels(X_train_c, lengths_train)
                class_counts = np.bincount(seq_labels, minlength=2).astype(np.float64)
                class_counts[class_counts == 0.0] = 1.0
                inv_freq = 1.0 / class_counts
                sample_weights = inv_freq[seq_labels]
                sampler = torch.utils.data.WeightedRandomSampler(
                    weights=torch.as_tensor(sample_weights, dtype=torch.double),
                    num_samples=len(sample_weights),
                    replacement=True
                )
                train_loader = torch.utils.data.DataLoader(
                    train_dataset,
                    batch_size=TRAINING_CONFIG['batch_size'],
                    sampler=sampler,
                    num_workers=TRAINING_CONFIG.get('num_workers', 4),
                    pin_memory=TRAINING_CONFIG.get('pin_memory', True)
                )
                print("Using weighted sampler for class balance")
            else:
                train_loader = torch.utils.data.DataLoader(
                    train_dataset,
                    batch_size=TRAINING_CONFIG['batch_size'],
                    shuffle=True,
                    num_workers=TRAINING_CONFIG.get('num_workers', 4),
                    pin_memory=TRAINING_CONFIG.get('pin_memory', True)
                )
            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=TRAINING_CONFIG['batch_size'],
                shuffle=False,
                num_workers=TRAINING_CONFIG.get('num_workers', 4),
                pin_memory=TRAINING_CONFIG.get('pin_memory', True)
            )
            print(f"Using torch DataLoader (num_workers={TRAINING_CONFIG.get('num_workers',4)}, pin_memory={TRAINING_CONFIG.get('pin_memory',True)})")
        except Exception as e:
            print(f"Warning: Could not use torch DataLoader, falling back to numpy loader: {str(e)[:120]}")
            train_loader = DataLoader(
                X_train_q, X_train_c, X_train_t, X_train_i, lengths_train, y_train,
                batch_size=TRAINING_CONFIG['batch_size'], shuffle=True
            )
            val_loader = DataLoader(
                X_val_q, X_val_c, X_val_t, X_val_i, lengths_val, y_val,
                batch_size=TRAINING_CONFIG['batch_size'], shuffle=False
            )
    else:
        train_loader = DataLoader(
            X_train_q, X_train_c, X_train_t, X_train_i, lengths_train, y_train,
            batch_size=TRAINING_CONFIG['batch_size'], shuffle=True
        )
        val_loader = DataLoader(
            X_val_q, X_val_c, X_val_t, X_val_i, lengths_val, y_val,
            batch_size=TRAINING_CONFIG['batch_size'], shuffle=False
        )
    
    # Initialize trainer
    print("\n" + "=" * 60)
    print("STEP 3: Building Model")
    print("=" * 60)
    
    trainer = DKTTrainer(device=args.device)
    trainer.build_model(num_questions, num_topics=num_topics)
    if args.init_checkpoint:
        print(f"Loading pretrained weights from: {args.init_checkpoint}")
        trainer.load_pretrained_weights(args.init_checkpoint)
        if args.freeze_encoder_epochs == 0 and TRAINING_CONFIG['data_source'] == 'primary':
            args.freeze_encoder_epochs = 5
            print("Auto fine-tune schedule enabled for primary data: freezing encoder for 5 epochs")
    trainer.set_finetune_schedule(args.freeze_encoder_epochs)
    max_pos_weight = float(TRAINING_CONFIG.get('max_pos_weight', 5.0))
    pos_weight = DKTTrainer._estimate_pos_weight(X_train_c, lengths_train, max_pos_weight=max_pos_weight)
    print(f"Estimated class balance: pos_weight={pos_weight:.4f}")
    trainer.setup_optimizer(
        pos_weight=pos_weight,
        encoder_lr_multiplier=args.encoder_lr_multiplier if args.init_checkpoint else 1.0,
        head_lr_multiplier=args.head_lr_multiplier if args.init_checkpoint else 1.0,
    )
    # Enable cuDNN benchmark for potential GPU performance improvements
    if trainer.device == 'cuda' and TRAINING_CONFIG.get('cudnn_benchmark', True):
        try:
            torch.backends.cudnn.benchmark = True
            print("cuDNN benchmark enabled")
        except Exception:
            pass
    
    # Train model
    print("\n" + "=" * 60)
    print("STEP 4: Training Model")
    print("=" * 60)
    
    trainer.train(
        train_loader,
        val_loader,
        num_epochs=args.epochs,
        early_stopping_patience=TRAINING_CONFIG['early_stopping_patience']
    )
    
    # Save results
    print("\n" + "=" * 60)
    print("STEP 5: Saving Results")
    print("=" * 60)
    
    history_path = f"{PATHS['results_dir']}/training_history.json"
    trainer.save_history(history_path)
    
    # Display accumulated history stats
    history_stats = DKTTrainer.get_accumulated_history_stats(history_path)
    if history_stats:
        print(f"\n📊 ACCUMULATED HISTORY STATISTICS:")
        print(f"   Total training runs: {history_stats['total_runs']}")
        print(f"   Total epochs across all runs: {history_stats['total_epochs']}")
        for i, ts in enumerate(history_stats['run_timestamps'], 1):
            print(f"   Run {i}: {ts}")

    checkpoint_metric = str(TRAINING_CONFIG.get('checkpoint_metric', 'val_auc')).lower()
    if checkpoint_metric == 'val_loss':
        best_idx = int(np.argmin(np.array(trainer.history['val_loss'])))
    else:
        best_idx = int(np.nanargmax(np.array(trainer.history['val_auc'])))
    best_metrics = {
        'epoch': best_idx + 1,
        'train_loss': float(trainer.history['train_loss'][best_idx]),
        'val_loss': float(trainer.history['val_loss'][best_idx]),
        'train_acc': float(trainer.history['train_acc'][best_idx]),
        'val_acc': float(trainer.history['val_acc'][best_idx]),
        'train_auc': float(trainer.history['train_auc'][best_idx]),
        'val_auc': float(trainer.history['val_auc'][best_idx]),
        'train_precision': float(trainer.history['train_precision'][best_idx]),
        'val_precision': float(trainer.history['val_precision'][best_idx]),
        'train_recall': float(trainer.history['train_recall'][best_idx]),
        'val_recall': float(trainer.history['val_recall'][best_idx]),
        'decision_threshold': float(trainer.history['decision_threshold'][best_idx]) if trainer.history['decision_threshold'] else float(trainer.decision_threshold)
    }

    final_metrics = {
        'epoch': len(trainer.history['val_loss']),
        'train_loss': float(trainer.history['train_loss'][-1]),
        'val_loss': float(trainer.history['val_loss'][-1]),
        'train_acc': float(trainer.history['train_acc'][-1]),
        'val_acc': float(trainer.history['val_acc'][-1]),
        'train_auc': float(trainer.history['train_auc'][-1]),
        'val_auc': float(trainer.history['val_auc'][-1]),
        'train_precision': float(trainer.history['train_precision'][-1]),
        'val_precision': float(trainer.history['val_precision'][-1]),
        'train_recall': float(trainer.history['train_recall'][-1]),
        'val_recall': float(trainer.history['val_recall'][-1]),
        'decision_threshold': float(trainer.history['decision_threshold'][-1]) if trainer.history['decision_threshold'] else float(trainer.decision_threshold)
    }
    
    # Build summary
    summary = {
        'num_questions': num_questions,
        'num_students': len(set(student_ids)),
        'total_interactions': len(df_assistments),
        'num_sequences': len(question_seqs),
        'split_strategy': 'temporal_per_student_non_overlapping_windows',
        'model_config': MODEL_CONFIG,
        'training_config': TRAINING_CONFIG,
        'data_config': DATA_CONFIG,
        'best_metrics': best_metrics,
        'final_metrics': final_metrics
    }
    
    # Save preprocessor for later use
    import pickle
    with open(f"{PATHS['results_dir']}/preprocessor.pkl", 'wb') as f:
        pickle.dump(preprocessor, f)
    print(f"Preprocessor saved")
    
    # Save summary (append to JSONL for accumulation)
    summary_path = Path(PATHS['results_dir']) / 'training_summary.jsonl'
    with open(summary_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(summary) + '\n')
    print(f"✓ Training summary appended to: {summary_path}")

    experiment_log = {
        'run_timestamp': datetime.now().isoformat(timespec='seconds'),
        'model_checkpoint': PATHS['model_checkpoint'],
        'results_dir': PATHS['results_dir'],
        'split_strategy': summary['split_strategy'],
        'num_questions': num_questions,
        'num_students': len(set(student_ids)),
        'total_interactions': len(df_assistments),
        'num_sequences': len(question_seqs),
        'best_metrics': best_metrics,
        'final_metrics': final_metrics,
        'model_config': MODEL_CONFIG,
        'training_config': TRAINING_CONFIG,
        'data_config': DATA_CONFIG
    }

    experiment_log_path = Path(PATHS['results_dir']) / 'experiments_log.jsonl'
    with open(experiment_log_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(experiment_log) + '\n')
    print(f"✓ Experiment log appended to: {experiment_log_path}")
    
    # ✅ AUTO-GENERATE PLOTS
    print("\n" + "=" * 80)
    print("STEP 6: Generating Training Plots")
    print("=" * 80)
    
    try:
        import subprocess
        train_dir = Path(__file__).parent
        plots_dir = Path(PATHS['results_dir']) / 'plots'
        
        result = subprocess.run(
            ['python', 'plot_training_history.py'],
            cwd=str(train_dir),
            capture_output=True,
            text=True,
            timeout=120
        )
        
        # Check if plots were actually generated (regardless of return code)
        if plots_dir.exists() and any(plots_dir.glob('*.png')):
            print("✓ Plots generated successfully!")
            # Count generated plots
            plot_files = list(plots_dir.glob('*.png'))
            latest_plots = sorted(plot_files, key=lambda p: p.stat().st_mtime, reverse=True)[:10]
            for plot_file in latest_plots:
                print(f"  ✓ {plot_file.name}")
        elif result.returncode == 0:
            print("✓ Plots generated successfully!")
            if result.stdout:
                for line in result.stdout.strip().split('\n')[-5:]:  # Last 5 lines
                    print(f"  {line}")
        else:
            print(f"⚠️  Warning: Could not verify plot generation")
            if result.returncode != 0 and result.stderr:
                # Only print first line of error
                first_error_line = result.stderr.split('\n')[0]
                print(f"  ({first_error_line[:100]}...)")
            print(f"  You can run manually: python plot_training_history.py")
    
    except subprocess.TimeoutExpired:
        print(f"⚠️  Warning: Plotting timed out (took > 120s)")
        print(f"  You can run manually: python plot_training_history.py")
    except Exception as e:
        print(f"⚠️  Warning: Could not auto-generate plots: {str(e)[:100]}")
        print(f"  You can run manually: python plot_training_history.py")
    
    print("\n" + "=" * 80)
    print("🎉 TRAINING COMPLETE!")
    print("=" * 80)
    
    # Display best metrics
    print(f"\n📊 BEST MODEL METRICS (Epoch {best_metrics['epoch']}):")
    print(f"   Train Loss: {best_metrics['train_loss']:.4f} | Val Loss: {best_metrics['val_loss']:.4f}")
    print(f"   Train Acc:  {best_metrics['train_acc']:.4f} | Val Acc:  {best_metrics['val_acc']:.4f}")
    print(f"   Train AUC:  {best_metrics['train_auc']:.4f} | Val AUC:  {best_metrics['val_auc']:.4f}")
    print(f"   Train Prec: {best_metrics['train_precision']:.4f} | Val Prec: {best_metrics['val_precision']:.4f}")
    print(f"   Train Rec:  {best_metrics['train_recall']:.4f} | Val Rec:  {best_metrics['val_recall']:.4f}")
    print(f"   Threshold:  {best_metrics['decision_threshold']:.4f}")
    
    # Display final metrics
    print(f"\n📈 FINAL METRICS (Epoch {final_metrics['epoch']}):")
    print(f"   Train Loss: {final_metrics['train_loss']:.4f} | Val Loss: {final_metrics['val_loss']:.4f}")
    print(f"   Train Acc:  {final_metrics['train_acc']:.4f} | Val Acc:  {final_metrics['val_acc']:.4f}")
    print(f"   Train AUC:  {final_metrics['train_auc']:.4f} | Val AUC:  {final_metrics['val_auc']:.4f}")
    print(f"   Train Prec: {final_metrics['train_precision']:.4f} | Val Prec: {final_metrics['val_precision']:.4f}")
    print(f"   Train Rec:  {final_metrics['train_recall']:.4f} | Val Rec:  {final_metrics['val_recall']:.4f}")
    print(f"   Threshold:  {final_metrics['decision_threshold']:.4f}")
    
    print(f"\n💾 SAVED FILES:")
    print(f"   ✓ Model checkpoint: {PATHS['model_checkpoint']}")
    print(f"   ✓ Results dir: {PATHS['results_dir']}")
    print(f"   ✓ Training history: {PATHS['results_dir']}/training_history.json")
    print(f"   ✓ Training summary: {PATHS['results_dir']}/training_summary.json")
    print(f"   ✓ Preprocessor: {PATHS['results_dir']}/preprocessor.pkl")
    
    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
