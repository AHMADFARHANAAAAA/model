"""
Deep Knowledge Tracing Configuration
Configuration parameters for model training and evaluation
"""

# Model Architecture
MODEL_CONFIG = {
    'input_size': 1,  # Single value: is_correct (0 or 1)
    'hidden_size': 128,  # LSTM hidden state dimension
    'num_layers': 2,  # Number of LSTM layers
    'output_size': 1,  # Output: mastery probability
    'dropout': 0.3,  # Dropout rate
}

# Training Parameters
TRAINING_CONFIG = {
    'batch_size': 32,
    'learning_rate': 0.001,
    'num_epochs': 50,
    'validation_split': 0.2,
    'early_stopping_patience': 10,
    'weight_decay': 1e-5,
    'device': 'cuda',  # Forced to use CUDA (GPU)
}

# Data Processing
DATA_CONFIG = {
    'max_sequence_length': 100,  # Maximum interactions per student
    'min_interactions': 5,  # Minimum interactions to include student
    'question_id_offset': 1,  # Offset for question IDs
    'random_seed': 42,
}

# Paths
PATHS = {
    'assistments_data': 'data/assistments_2009_2010.csv',
    'primary_data': 'data/primary_interactions.csv',
    'model_checkpoint': 'checkpoints/dkt_model.pth',
    'results_dir': 'results/',
}

# Recommendation Parameters
RECOMMENDATION_CONFIG = {
    'mastery_threshold': 0.7,  # Threshold for considering concept mastered
    'low_performance_threshold': 0.4,  # Threshold for weak concepts
    'top_recommendations': 3,  # Number of recommendations to generate
}
