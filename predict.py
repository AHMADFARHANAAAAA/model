"""
Prediction Interface for DKT Model
Generates recommendations based on model predictions
"""

import torch
import numpy as np
import pandas as pd
import pickle
import json
from typing import Dict, List, Tuple
from pathlib import Path

from dkt_model import DKTModel
from data_preprocessor import DataPreprocessor
from config import MODEL_CONFIG, TRAINING_CONFIG, DATA_CONFIG, PATHS, RECOMMENDATION_CONFIG


class DKTPredictor:
    """
    Prediction interface for DKT model
    Generates mastery predictions and recommendations
    """
    
    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = None
        self.preprocessor = None
        self.summary = None
        self._load_resources()
    
    def _load_resources(self):
        """Load model, preprocessor, and configuration"""
        # Load summary
        try:
            with open(f"{PATHS['results_dir']}/training_summary.json", 'r') as f:
                self.summary = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError("Training summary not found. Please train the model first.")
        
        # Load preprocessor
        try:
            with open(f"{PATHS['results_dir']}/preprocessor.pkl", 'rb') as f:
                self.preprocessor = pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError("Preprocessor not found. Please train the model first.")
        
        # Load model
        num_questions = self.summary['num_questions']
        self.model = DKTModel(
            num_questions=num_questions,
            hidden_size=MODEL_CONFIG['hidden_size'],
            num_layers=MODEL_CONFIG['num_layers'],
            embedding_size=50,
            dropout=MODEL_CONFIG['dropout'],
            output_size=MODEL_CONFIG['output_size']
        ).to(self.device)
        
        checkpoint = torch.load(PATHS['model_checkpoint'], map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print("Model resources loaded successfully!")
    
    def predict_mastery(
        self,
        student_interactions: List[Tuple[int, int]]
    ) -> Dict[str, float]:
        """
        Predict mastery for each concept based on student interactions
        
        Args:
            student_interactions: List of (question_id, is_correct) tuples
                                 Question IDs should be from the dataset
        
        Returns:
            Dictionary of {concept_id: mastery_probability}
        
        Example:
            interactions = [(1, 1), (5, 0), (2, 1), (3, 1)]
            mastery = predictor.predict_mastery(interactions)
            # Output: {
            #     'topic_1': 0.35,
            #     'topic_2': 0.82,
            #     'topic_3': 0.12
            # }
        """
        if len(student_interactions) < DATA_CONFIG['min_interactions']:
            raise ValueError(
                f"Minimum {DATA_CONFIG['min_interactions']} interactions required "
                f"for prediction. Got {len(student_interactions)}"
            )
        
        # Extract question IDs and correctness
        question_ids, correctness = zip(*student_interactions)
        
        # Encode question IDs
        try:
            encoded_ids = [self.preprocessor.question_to_id[q] for q in question_ids]
        except KeyError as e:
            raise ValueError(f"Unknown question ID: {e}")
        
        # Prepare tensors
        q_tensor = torch.tensor([encoded_ids], dtype=torch.long).to(self.device)
        c_tensor = torch.tensor([correctness], dtype=torch.long).to(self.device)
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            output, _ = self.model(q_tensor, c_tensor)  # (1, seq_len, 1)
            mastery_score = output.mean().item()
        
        # Return mastery for each concept
        # For simplicity, we return overall mastery
        # In a real system, you'd track mastery per concept
        masteries = {
            'overall': mastery_score,
            'trend': self._calculate_trend(correctness)
        }
        
        return masteries
    
    def predict_next_performance(
        self,
        student_interactions: List[Tuple[int, int]],
        next_question_id: int
    ) -> float:
        """
        Predict probability of answering next question correctly
        
        Args:
            student_interactions: Student's interaction history
            next_question_id: Question ID to predict performance on
        
        Returns:
            Probability of correctness (0-1)
        """
        if next_question_id not in self.preprocessor.question_to_id:
            raise ValueError(f"Unknown question ID: {next_question_id}")
        
        # Get encoded ID
        encoded_id = self.preprocessor.question_to_id[next_question_id]
        
        # Prepare history
        question_ids, correctness = zip(*student_interactions)
        try:
            encoded_ids = [self.preprocessor.question_to_id[q] for q in question_ids]
        except KeyError as e:
            raise ValueError(f"Unknown question ID: {e}")
        
        # Add next question (we'll predict its correctness)
        encoded_ids.append(encoded_id)
        correctness = list(correctness) + [0]  # Placeholder
        
        # Prepare tensors
        q_tensor = torch.tensor([encoded_ids], dtype=torch.long).to(self.device)
        c_tensor = torch.tensor([correctness], dtype=torch.long).to(self.device)
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            output, _ = self.model(q_tensor, c_tensor)  # (1, seq_len, 1)
            next_performance = output[0, -1, 0].item()  # Last prediction
        
        return next_performance
    
    def generate_recommendations(
        self,
        student_interactions: List[Tuple[int, int]],
        available_questions: Dict[int, str]
    ) -> Dict[str, List]:
        """
        Generate learning recommendations based on mastery predictions
        
        Args:
            student_interactions: Student's interaction history
            available_questions: Dict of {question_id: question_topic}
        
        Returns:
            Dictionary with:
            - 'weak_areas': List of concepts to focus on
            - 'recommended_questions': List of recommended question IDs
            - 'mastery_levels': Dict of mastery per concept
        
        Example:
            recommendations = predictor.generate_recommendations(
                interactions,
                {1: 'Variables', 2: 'Functions', 3: 'Loops'}
            )
        """
        # Predict overall mastery
        masteries = self.predict_mastery(student_interactions)
        overall_mastery = masteries['overall']
        
        # Identify weak areas
        weak_areas = []
        if overall_mastery < RECOMMENDATION_CONFIG['low_performance_threshold']:
            weak_areas.append({
                'area': 'Overall',
                'mastery': overall_mastery,
                'recommendation': 'Focus on fundamental concepts'
            })
        
        # Identify recommended questions based on predicted performance
        recommendations = []
        for q_id, topic in available_questions.items():
            try:
                pred_performance = self.predict_next_performance(student_interactions, q_id)
                
                # Recommend questions with moderate difficulty
                # (where student has 30-70% probability of success)
                if 0.3 < pred_performance < 0.7:
                    recommendations.append({
                        'question_id': q_id,
                        'topic': topic,
                        'predicted_performance': pred_performance,
                        'difficulty': 'Medium'
                    })
            except ValueError:
                continue
        
        # Sort by predicted performance (ascending - harder questions first)
        recommendations.sort(key=lambda x: x['predicted_performance'])
        
        # Get top recommendations
        top_recommendations = recommendations[:RECOMMENDATION_CONFIG['top_recommendations']]
        
        return {
            'overall_mastery': overall_mastery,
            'mastery_trend': masteries['trend'],
            'weak_areas': weak_areas,
            'recommended_questions': top_recommendations,
            'recommendation_count': len(top_recommendations),
            'mastery_level': self._get_mastery_level(overall_mastery)
        }
    
    def batch_predict(self, students_data: pd.DataFrame) -> pd.DataFrame:
        """
        Batch prediction for multiple students
        
        Args:
            students_data: DataFrame with columns:
                - student_id
                - question_id
                - is_correct
                - timestamp
        
        Returns:
            DataFrame with mastery predictions per student
        """
        results = []
        
        for student_id, group in students_data.groupby('student_id'):
            if len(group) < DATA_CONFIG['min_interactions']:
                continue
            
            # Sort by timestamp
            group = group.sort_values('timestamp')
            interactions = list(zip(group['question_id'], group['is_correct']))
            
            try:
                masteries = self.predict_mastery(interactions)
                results.append({
                    'student_id': student_id,
                    'num_interactions': len(interactions),
                    'overall_mastery': masteries['overall'],
                    'mastery_trend': masteries['trend']
                })
            except (ValueError, KeyError):
                continue
        
        return pd.DataFrame(results)
    
    @staticmethod
    def _calculate_trend(correctness: Tuple[int, ...]) -> str:
        """
        Calculate learning trend from correctness sequence
        
        Returns:
            'improving', 'declining', or 'stable'
        """
        if len(correctness) < 5:
            return 'insufficient_data'
        
        first_half = np.mean(correctness[:len(correctness)//2])
        second_half = np.mean(correctness[len(correctness)//2:])
        
        diff = second_half - first_half
        
        if diff > 0.1:
            return 'improving'
        elif diff < -0.1:
            return 'declining'
        else:
            return 'stable'
    
    @staticmethod
    def _get_mastery_level(mastery: float) -> str:
        """Convert mastery score to level"""
        if mastery >= RECOMMENDATION_CONFIG['mastery_threshold']:
            return 'Mastered'
        elif mastery >= 0.5:
            return 'Intermediate'
        elif mastery >= 0.3:
            return 'Beginner'
        else:
            return 'Novice'


def example_usage():
    """
    Example usage of DKTPredictor
    """
    # Initialize predictor
    predictor = DKTPredictor()
    
    # Example student interactions
    student_interactions = [
        (1, 1),   # Question 1, Correct
        (5, 0),   # Question 5, Incorrect
        (2, 1),   # Question 2, Correct
        (3, 1),   # Question 3, Correct
        (4, 0),   # Question 4, Incorrect
        (6, 1),   # Question 6, Correct
    ]
    
    # Predict mastery
    print("Predicting mastery...")
    masteries = predictor.predict_mastery(student_interactions)
    print(f"Overall Mastery: {masteries['overall']:.4f}")
    print(f"Trend: {masteries['trend']}")
    
    # Predict next question performance
    print("\nPredicting next question performance...")
    next_perf = predictor.predict_next_performance(student_interactions, 7)
    print(f"Predicted performance on Q7: {next_perf:.4f}")
    
    # Generate recommendations
    print("\nGenerating recommendations...")
    available_questions = {
        7: 'Variables',
        8: 'Functions',
        9: 'Loops',
        10: 'Arrays'
    }
    
    recommendations = predictor.generate_recommendations(
        student_interactions,
        available_questions
    )
    
    print(f"Overall Mastery: {recommendations['overall_mastery']:.4f}")
    print(f"Mastery Level: {recommendations['mastery_level']}")
    print(f"\nRecommended Questions:")
    for rec in recommendations['recommended_questions']:
        print(f"  - Q{rec['question_id']} ({rec['topic']}): {rec['predicted_performance']:.4f}")


def main():
    """Main entry point"""
    example_usage()


if __name__ == '__main__':
    main()
