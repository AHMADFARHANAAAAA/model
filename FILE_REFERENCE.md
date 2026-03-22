"""
DKT_Model Project - Complete File Reference
Generated: March 2026

This document provides a quick reference for all files in the DKT_Model project.
"""

# ==============================================================================
# CORE MODEL FILES
# ==============================================================================

DKTMODEL_PY = """
File: dkt_model.py
================================================================================
Purpose: Deep Knowledge Tracing model architecture implementation

Key Classes:
  - DKTModel: Main LSTM-based model
  - DKTLoss: Custom loss function for DKT training

Features:
  - Question embedding layer (maps question IDs to vectors)
  - Correctness embedding layer (maps 0/1 to vectors)
  - 2-layer LSTM for sequence processing
  - Dense output layers with sigmoid activation
  - Methods for inference and prediction

Usage:
  from dkt_model import DKTModel
  model = DKTModel(num_questions=189, hidden_size=128)
  output, hidden = model(question_ids, correctness)

Dependencies: torch, numpy
Lines: ~300
"""

DATA_PREPROCESSOR_PY = """
File: data_preprocessor.py
================================================================================
Purpose: Data loading, preprocessing, and sequence generation

Key Classes:
  - DataPreprocessor: Main preprocessing pipeline
  - DataLoader: PyTorch-style batch iterator

Features:
  - Load ASSISTments and primary datasets
  - Flexible column name mapping
  - Question ID encoding
  - Sequence creation and padding
  - Train/val/test splitting
  - Batch iteration with padding support

Usage:
  preprocessor = DataPreprocessor()
  df = preprocessor.load_assistments_data('data/assistments.csv')
  seqs_q, seqs_c, ids = preprocessor.create_sequences(df)
  X_q, X_c, lengths = preprocessor.prepare_training_data(seqs_q, seqs_c)

Dependencies: pandas, numpy, sklearn
Lines: ~450
"""

CONFIG_PY = """
File: config.py
================================================================================
Purpose: Central configuration for model and training

Contains:
  - MODEL_CONFIG: Model hyperparameters (hidden_size, num_layers, etc)
  - TRAINING_CONFIG: Training parameters (batch_size, learning_rate, etc)
  - DATA_CONFIG: Data processing parameters (max_sequence_length, etc)
  - PATHS: File paths for data and models
  - RECOMMENDATION_CONFIG: Recommendation thresholds

Usage:
  from config import MODEL_CONFIG, TRAINING_CONFIG
  model = DKTModel(**MODEL_CONFIG)

Dependencies: None
Lines: ~50
"""

# ==============================================================================
# TRAINING & EVALUATION
# ==============================================================================

TRAIN_PY = """
File: train.py
================================================================================
Purpose: Complete training pipeline for DKT model

Main Function:
  - Loads ASSISTments dataset
  - Creates and trains DKT model
  - Implements early stopping
  - Saves best checkpoint

Key Class:
  - DKTTrainer: Handles training loop, validation, checkpointing

Usage:
  python train.py

Outputs:
  - checkpoints/dkt_model.pth (trained model)
  - results/training_history.json (loss/accuracy curves)
  - results/training_summary.json (metadata)
  - results/preprocessor.pkl (for inference)

Training Time: 30-60 min (30 on GPU, 60 on CPU)
Dependencies: torch, pandas, numpy, tqdm
Lines: ~380
"""

EVALUATE_PY = """
File: evaluate.py
================================================================================
Purpose: Model evaluation on test dataset

Main Functions:
  - Loads trained model
  - Loads primary dataset
  - Computes metrics: accuracy, precision, recall, AUC
  - Saves evaluation results

Key Class:
  - DKTEvaluator: Handles model evaluation

Metrics Computed:
  - Accuracy: (TP + TN) / Total
  - Precision: TP / (TP + FP)
  - Recall: TP / (TP + FN)
  - AUC: Area under ROC curve
  - Confusion Matrix

Usage:
  python evaluate.py

Outputs:
  - results/evaluation_metrics.json (metrics and confusion matrix)

Requires:
  - checkpoints/dkt_model.pth (from training)
  - data/primary_interactions.csv (test data)

Dependencies: torch, pandas, numpy, sklearn
Lines: ~350
"""

# ==============================================================================
# PREDICTION & RECOMMENDATIONS
# ==============================================================================

PREDICT_PY = """
File: predict.py
================================================================================
Purpose: Generate predictions and recommendations

Main Class:
  - DKTPredictor: Load model and generate recommendations

Key Methods:
  - predict_mastery(): Predict overall mastery (0-1)
  - predict_next_performance(): Predict performance on specific question
  - generate_recommendations(): Recommend next questions to learn
  - batch_predict(): Predict for multiple students

Features:
  - Loads trained model automatically
  - Calculates learning trend (improving/declining/stable)
  - Generates difficulty-appropriate recommendations
  - Works with variable-length student histories

Usage:
  from predict import DKTPredictor
  
  predictor = DKTPredictor()
  
  # Predict mastery
  mastery = predictor.predict_mastery([(1,1), (5,0), (2,1)])
  
  # Generate recommendations
  recs = predictor.generate_recommendations(
      interactions,
      {7: 'Variables', 8: 'Functions'}
  )

Requires:
  - checkpoints/dkt_model.pth (trained model)
  - results/preprocessor.pkl (question mappings)

Dependencies: torch, numpy, pandas
Lines: ~450
"""

# ==============================================================================
# UTILITIES & HELPERS
# ==============================================================================

EXPORT_DATA_PY = """
File: export_data.py
================================================================================
Purpose: Export interaction data from web system to CSV

Main Functions:
  - export_interactions_to_csv(): Export from Prisma database
  - export_assistments_data(): Setup ASSISTments dataset
  - verify_data_structure(): Validate CSV format

Features:
  - Handles multiple database backends (Prisma, SQLite, PostgreSQL)
  - Data validation and structure checking
  - Column mapping for flexibility

Usage:
  python export_data.py

Creates:
  - data/primary_interactions.csv (from your web system)
  - data/assistments_2009_2010.csv (ASSISTments data)

Time: 2-5 minutes
Dependencies: pandas, subprocess
Lines: ~280
"""

QUICKSTART_PY = """
File: quickstart.py
================================================================================
Purpose: Interactive quick-start guide

Provides:
  - Setup verification (dependencies, data files)
  - Menu-driven interface for training/evaluation
  - Automatic pipeline execution
  - Error handling and diagnostics

Usage:
  # Interactive mode
  python quickstart.py
  
  # Automatic mode
  python quickstart.py --auto

Features:
  - Checks for required dependencies
  - Verifies data files exist
  - Runs training, evaluation, prediction sequentially
  - Provides feedback at each step

Modes:
  1. Interactive: User chooses which steps to run
  2. Automatic (--auto): Runs complete pipeline

Dependencies: torch, pandas, numpy
Lines: ~280
"""

# ==============================================================================
# DOCUMENTATION
# ==============================================================================

README_MD = """
File: README.md
================================================================================
Purpose: Complete technical documentation

Sections:
  1. Overview - Project goals and key features
  2. Installation - Dependency installation
  3. Data Preparation - CSV format and structure
  4. Configuration - Adjustable parameters
  5. Training Pipeline - Step-by-step guide
  6. Model Architecture - Layer descriptions
  7. Data Preprocessing - How to format data
  8. Training Details - Loss function, optimization, regularization
  9. Evaluation Metrics - Interpretation of metrics
  10. Troubleshooting - Common issues and solutions
  11. Advanced Features - Fine-tuning, custom concepts, multi-task learning
  12. Thesis Citation - How to cite this work
  13. References - Related papers and resources

Length: ~800 lines
Format: Markdown with code examples and mathematical notation
"""

INTEGRATION_GUIDE_MD = """
File: INTEGRATION_GUIDE.md
================================================================================
Purpose: Integrate DKT model with web system

Content:
  1. Data Export from Web System - SQL queries for all databases
  2. Python Backend Service - DKTRecommendationService class
  3. API Integration - Next.js API endpoint implementation
  4. Frontend Integration - Learning page UI updates
  5. Retraining Scheduler - Automated model updates
  6. Monitoring - Performance tracking and alerts
  7. Integration Checklist - Step-by-step checklist

Contains:
  - Code snippets in Python, Node.js, TypeScript
  - SQL queries for data export
  - Flask/FastAPI endpoint examples
  - Cron job configuration
  - Monitoring and logging setup

Length: ~600 lines
"""

QUICKSTART_TXT = """
File: QUICKSTART.txt
================================================================================
Purpose: Quick reference guide (plain text, not markdown)

Sections:
  1. Quick Start (5 Steps)
     - Step 1: Install dependencies
     - Step 2: Prepare data
     - Step 3: Train model
     - Step 4: Evaluate model
     - Step 5: Generate recommendations
  
  2. Integration with Web System
  3. File Structure
  4. Configuration Options
  5. Troubleshooting
  6. Understanding the Model
  7. Evaluation Metrics Explained
  8. Next Steps
  9. Useful Commands
  10. Support & Resources

Features:
  - Plain text format for easy reading
  - Time estimates for each step
  - Expected outputs at each stage
  - Common issues and solutions
  - Quick reference commands

Length: ~400 lines
Format: Plain text with ASCII formatting
"""

# ==============================================================================
# GENERATED FILES (CREATED DURING TRAINING)
# ==============================================================================

GENERATED_FILES = """
Generated During Training:
------------------------

checkpoints/dkt_model.pth
  - Format: PyTorch model checkpoint
  - Size: ~50-100 MB depending on model size
  - Contains: Model weights, optimizer state, training history
  - Created by: train.py
  - Used by: evaluate.py, predict.py

results/training_history.json
  - Format: JSON
  - Contains: Train/val loss and accuracy per epoch
  - Created by: train.py
  - Usage: Plot training curves

results/evaluation_metrics.json
  - Format: JSON
  - Contains: Accuracy, precision, recall, AUC, confusion matrix
  - Created by: evaluate.py
  - Usage: Report results in thesis

results/training_summary.json
  - Format: JSON
  - Contains: Model config, training params, data stats
  - Created by: train.py
  - Usage: Preserve experiment metadata

results/preprocessor.pkl
  - Format: Python pickle
  - Contains: Question ID mappings, preprocessing state
  - Created by: train.py
  - Used by: evaluate.py, predict.py

data/primary_interactions.csv
  - Format: CSV with headers
  - Columns: student_id, question_id, topic_id, is_correct, 
            attempt_order, timestamp, response_time
  - Created by: export_data.py
  - Source: Your web system database

data/assistments_2009_2010.csv
  - Format: CSV with headers
  - Size: ~500MB
  - Contains: 500,000+ interactions
  - Columns: student_id, question_id, correct, timestamp
  - Source: Copied from workspace by export_data.py
"""

# ==============================================================================
# PROJECT STATISTICS
# ==============================================================================

STATISTICS = """
Project Statistics:
==================

Total Files: 10 core files + generated outputs
Total Code Lines: ~2,500 (excluding documentation)
Total Documentation: ~1,600 lines

Breakdown by Purpose:
  - Model Architecture: 300 lines (dkt_model.py)
  - Data Processing: 450 lines (data_preprocessor.py)
  - Training: 380 lines (train.py)
  - Evaluation: 350 lines (evaluate.py)
  - Prediction: 450 lines (predict.py)
  - Utilities: 560 lines (export_data.py, quickstart.py, config.py)
  - Documentation: 1,600 lines (README.md, INTEGRATION_GUIDE.md, QUICKSTART.txt)

Time to Implement:
  - Setup: 5-10 minutes
  - Data Preparation: 2-5 minutes
  - Training: 30-60 minutes (depending on hardware)
  - Evaluation: 5-10 minutes
  - Integration: 2-4 hours

Hardware Requirements:
  - Minimum: CPU only (50+ MB RAM required)
  - Recommended: NVIDIA GPU with CUDA 11.0+
  - Storage: 1GB for code + models + data

Python Package Size:
  - PyTorch: 500+ MB
  - Dependencies: 200+ MB
  - Total: ~700-800 MB
"""

# ==============================================================================
# QUICK REFERENCE TABLE
# ==============================================================================

print("""
╔════════════════════════════════════════════════════════════════════════════╗
║           DKT_MODEL PROJECT - QUICK REFERENCE                             ║
╠════════════════════════════════════════════════════════════════════════════╣
║                                                                            ║
║  CORE MODEL FILES                                                          ║
║  ├─ dkt_model.py ................. LSTM model architecture (300 lines)    ║
║  ├─ data_preprocessor.py ......... Data loading & preprocessing (450)    ║
║  └─ config.py .................... Configuration parameters (50)         ║
║                                                                            ║
║  TRAINING & EVALUATION                                                     ║
║  ├─ train.py ..................... Training pipeline (380 lines)          ║
║  └─ evaluate.py .................. Evaluation script (350 lines)          ║
║                                                                            ║
║  INFERENCE & RECOMMENDATIONS                                              ║
║  └─ predict.py ................... Prediction interface (450 lines)       ║
║                                                                            ║
║  UTILITIES                                                                 ║
║  ├─ export_data.py ............... Data export helper (280 lines)        ║
║  └─ quickstart.py ................ Interactive guide (280 lines)         ║
║                                                                            ║
║  DOCUMENTATION                                                             ║
║  ├─ README.md .................... Full documentation (~800 lines)       ║
║  ├─ INTEGRATION_GUIDE.md ......... Integration steps (~600 lines)       ║
║  └─ QUICKSTART.txt ............... Quick reference (~400 lines)          ║
║                                                                            ║
║  QUICK START COMMANDS                                                      ║
║  ├─ pip install -r requirements.txt
║  ├─ python export_data.py
║  ├─ python DKT_Model/train.py
║  ├─ python DKT_Model/evaluate.py
║  └─ python DKT_Model/predict.py
║                                                                            ║
║  TYPICAL WORKFLOW                                                          ║
║  1. Install dependencies (5 min)
║  2. Export data (2 min)
║  3. Train model (30-60 min)
║  4. Evaluate model (5 min)
║  5. Integrate & deploy (2-4 hours)
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝
""")

# ==============================================================================
# GETTING HELP
# ==============================================================================

HELP = """
Getting Help:
=============

1. Read QUICKSTART.txt first (this file is the entry point)
2. Check README.md for detailed technical documentation
3. Look at INTEGRATION_GUIDE.md for web system integration
4. Run quickstart.py for interactive guidance
5. Check troubleshooting sections in README.md

Common Tasks:

  Train the model:
    python DKT_Model/train.py
  
  Evaluate the model:
    python DKT_Model/evaluate.py
  
  Generate recommendations:
    python DKT_Model/predict.py
  
  Export data from web system:
    python DKT_Model/export_data.py
  
  Interactive mode:
    python DKT_Model/quickstart.py
  
  View training curves:
    python -c "import json; import matplotlib.pyplot as plt; \\
      h = json.load(open('DKT_Model/results/training_history.json')); \\
      plt.plot(h['train_loss'], label='Train'); \\
      plt.plot(h['val_loss'], label='Val'); plt.legend(); plt.show()"

File Navigation:

  All source code: DKT_Model/
  Input data: DKT_Model/data/
  Output models: DKT_Model/checkpoints/
  Results: DKT_Model/results/
  Logs: DKT_Model/logs/ (created when running)

For More Help:
  - PyTorch: https://pytorch.org
  - LSTM Guide: http://colah.github.io/posts/2015-08-Understanding-LSTMs/
  - DKT Paper: https://arxiv.org/abs/1506.05908
"""
