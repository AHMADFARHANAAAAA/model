"""
Data Preprocessing Pipeline for DKT
Handles loading and processing both ASSISTments and primary datasets
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict
from sklearn.preprocessing import StandardScaler
import os


class DataPreprocessor:
    """
    Handles data loading, preprocessing, and sequence generation
    for Deep Knowledge Tracing
    """
    
    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        np.random.seed(random_seed)
        self.question_to_id = {}
        self.topic_to_id = {}
        
    def load_assistments_data(self, filepath: str) -> pd.DataFrame:
        """
        Load and preprocess ASSISTments dataset
        
        Expected columns:
        - student_id
        - question_id (or skill_id)
        - correct
        
        Args:
            filepath: Path to ASSISTments CSV file
        
        Returns:
            Preprocessed DataFrame with columns:
            - student_id, question_id, is_correct, timestamp
        """
        print(f"Loading ASSISTments data from {filepath}...")
        df = pd.read_csv(filepath)
        
        # Column name mapping for flexibility
        column_mapping = {
            'Outcome': 'is_correct',
            'correct': 'is_correct',
            'KCs': 'topic_id',
            'Skill': 'topic_id',
            'student': 'student_id',
            'Student': 'student_id',
            'user_id': 'student_id',
            'problem_id': 'question_id',
            'Problem': 'question_id',
        }
        
        # Rename columns
        df = df.rename(columns=column_mapping)
        
        # Ensure required columns exist
        required_cols = ['student_id', 'question_id', 'is_correct']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Convert correctness to binary
        if df['is_correct'].dtype != 'int64':
            df['is_correct'] = (df['is_correct'] == 1).astype(int)
            # Try alternative success indicators
            if df['is_correct'].sum() == 0:
                df['is_correct'] = (df['is_correct'].isin(['correct', 'Correct', 'True', 1])).astype(int)
        
        # Add timestamp if not present
        if 'timestamp' not in df.columns:
            df['timestamp'] = pd.date_range(start='2020-01-01', periods=len(df), freq='1S')
        else:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Sort by student and timestamp
        df = df.sort_values(['student_id', 'timestamp']).reset_index(drop=True)
        
        # Add topic_id if not present
        if 'topic_id' not in df.columns:
            df['topic_id'] = 1  # Default topic
        
        print(f"Loaded {len(df)} interactions from {df['student_id'].nunique()} students")
        return df
    
    def load_primary_data(self, filepath: str) -> pd.DataFrame:
        """
        Load student interaction data from web system
        
        Expected columns:
        - student_id
        - question_id
        - topic_id
        - is_correct
        - timestamp
        
        Args:
            filepath: Path to primary dataset CSV
        
        Returns:
            Preprocessed DataFrame
        """
        print(f"Loading primary data from {filepath}...")
        df = pd.read_csv(filepath)
        
        # Ensure required columns
        required_cols = ['student_id', 'question_id', 'is_correct', 'timestamp']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Convert data types
        df['is_correct'] = df['is_correct'].astype(int)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Sort by student and timestamp
        df = df.sort_values(['student_id', 'timestamp']).reset_index(drop=True)
        
        print(f"Loaded {len(df)} interactions from {df['student_id'].nunique()} students")
        return df
    
    def build_question_map(self, df: pd.DataFrame) -> Dict[int, int]:
        """
        Build mapping from question IDs to sequential indices
        
        Args:
            df: DataFrame with question_id column
        
        Returns:
            Dictionary mapping question_id -> encoded_id
        """
        unique_questions = df['question_id'].unique()
        self.question_to_id = {q: i + 1 for i, q in enumerate(unique_questions)}
        return self.question_to_id
    
    def encode_questions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode question IDs to sequential indices"""
        df = df.copy()
        df['question_id_encoded'] = df['question_id'].map(self.question_to_id)
        return df
    
    def create_sequences(
        self,
        df: pd.DataFrame,
        max_length: int = 100,
        min_interactions: int = 5
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[int]]:
        """
        Convert student interactions into sequences for LSTM input
        
        Args:
            df: DataFrame with interactions
            max_length: Maximum sequence length
            min_interactions: Minimum interactions per student to include
        
        Returns:
            - question_sequences: List of (seq_len,) arrays with question IDs
            - correctness_sequences: List of (seq_len,) arrays with correctness
            - student_ids: List of student IDs
        """
        question_sequences = []
        correctness_sequences = []
        student_ids = []
        
        for student_id, group in df.groupby('student_id'):
            # Filter students with minimum interactions
            if len(group) < min_interactions:
                continue
            
            # Get sequences
            questions = group['question_id_encoded'].values
            correctness = group['is_correct'].values
            
            # Handle long sequences
            if len(questions) > max_length:
                # Split into overlapping windows
                for i in range(0, len(questions) - max_length + 1, max_length // 2):
                    q_seq = questions[i:i + max_length]
                    c_seq = correctness[i:i + max_length]
                    
                    question_sequences.append(q_seq)
                    correctness_sequences.append(c_seq)
                    student_ids.append(student_id)
            else:
                question_sequences.append(questions)
                correctness_sequences.append(correctness)
                student_ids.append(student_id)
        
        print(f"Created {len(question_sequences)} sequences from {len(set(student_ids))} students")
        return question_sequences, correctness_sequences, student_ids
    
    def prepare_training_data(
        self,
        question_sequences: List[np.ndarray],
        correctness_sequences: List[np.ndarray],
        max_length: int = 100
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for training with padding and masking
        
        Args:
            question_sequences: List of question sequences
            correctness_sequences: List of correctness sequences
            max_length: Pad sequences to this length
        
        Returns:
            - padded_questions: (N, max_length) array
            - padded_correctness: (N, max_length) array
            - sequence_lengths: (N,) array of actual lengths
        """
        N = len(question_sequences)
        
        # Initialize padded arrays
        padded_questions = np.zeros((N, max_length), dtype=np.int32)
        padded_correctness = np.zeros((N, max_length), dtype=np.int32)
        sequence_lengths = np.zeros(N, dtype=np.int32)
        
        # Fill arrays with padding
        for i, (q_seq, c_seq) in enumerate(zip(question_sequences, correctness_sequences)):
            length = min(len(q_seq), max_length)
            padded_questions[i, :length] = q_seq[:length]
            padded_correctness[i, :length] = c_seq[:length]
            sequence_lengths[i] = length
        
        return padded_questions, padded_correctness, sequence_lengths
    
    def split_train_val_test(
        self,
        X_questions: np.ndarray,
        X_correctness: np.ndarray,
        y: np.ndarray,
        train_ratio: float = 0.6,
        val_ratio: float = 0.2,
        test_ratio: float = 0.2
    ) -> Tuple:
        """
        Split data into train/val/test sets
        
        Args:
            X_questions: Question sequences
            X_correctness: Correctness sequences
            y: Target variable
            train_ratio: Train set ratio
            val_ratio: Validation set ratio
            test_ratio: Test set ratio
        
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        assert train_ratio + val_ratio + test_ratio == 1.0
        
        N = len(X_questions)
        indices = np.random.permutation(N)
        
        train_idx = int(N * train_ratio)
        val_idx = int(N * (train_ratio + val_ratio))
        
        train_indices = indices[:train_idx]
        val_indices = indices[train_idx:val_idx]
        test_indices = indices[val_idx:]
        
        print(f"Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")
        
        return (
            (X_questions[train_indices], X_correctness[train_indices]),
            (X_questions[val_indices], X_correctness[val_indices]),
            (X_questions[test_indices], X_correctness[test_indices]),
            y[train_indices],
            y[val_indices],
            y[test_indices]
        )
    
    @staticmethod
    def create_targets(correctness_sequences: List[np.ndarray]) -> np.ndarray:
        """
        Create target labels (correctness for next question)
        
        Args:
            correctness_sequences: List of correctness arrays
        
        Returns:
            (N,) array of average correctness (target)
        """
        targets = np.array([np.mean(c_seq) for c_seq in correctness_sequences])
        return targets


class DataLoader:
    """
    PyTorch-style DataLoader for batch iteration
    """
    
    def __init__(
        self,
        questions: np.ndarray,
        correctness: np.ndarray,
        lengths: np.ndarray,
        targets: np.ndarray,
        batch_size: int = 32,
        shuffle: bool = False
    ):
        self.questions = questions
        self.correctness = correctness
        self.lengths = lengths
        self.targets = targets
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_samples = len(questions)
        
        if shuffle:
            self.indices = np.random.permutation(self.n_samples)
        else:
            self.indices = np.arange(self.n_samples)
        
        self.current_idx = 0
    
    def __iter__(self):
        self.current_idx = 0
        if self.shuffle:
            self.indices = np.random.permutation(self.n_samples)
        return self
    
    def __next__(self):
        if self.current_idx >= self.n_samples:
            raise StopIteration
        
        batch_indices = self.indices[self.current_idx:self.current_idx + self.batch_size]
        
        batch = {
            'questions': self.questions[batch_indices],
            'correctness': self.correctness[batch_indices],
            'lengths': self.lengths[batch_indices],
            'targets': self.targets[batch_indices]
        }
        
        self.current_idx += self.batch_size
        return batch
    
    def __len__(self):
        return (self.n_samples + self.batch_size - 1) // self.batch_size
