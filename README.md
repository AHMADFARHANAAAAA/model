# Deep Knowledge Tracing (DKT) Model

A PyTorch-based implementation of Deep Knowledge Tracing for predicting student learning and generating personalized recommendations.

## Overview

This project implements a complete DKT system for your thesis:

1. **Model Architecture**: LSTM-based neural network that learns from sequential student interactions
2. **Pre-training**: Train on ASSISTments dataset (secondary data)
3. **Evaluation**: Test on your primary dataset from the web system
4. **Recommendations**: Generate personalized learning recommendations based on mastery predictions

### Key Features

- 🧠 LSTM-based DKT model with embedding layers
- 📊 Supports variable-length student sequences
- 🎯 Predicts knowledge mastery for each concept
- 📈 Generates personalized learning recommendations
- 📉 Evaluates with Accuracy, Precision, Recall, and AUC metrics
- 💾 Model checkpointing and training history

---

## Project Structure

```
DKT_Model/
├── config.py                 # Configuration parameters
├── dkt_model.py             # LSTM-based DKT model architecture
├── data_preprocessor.py     # Data loading and preprocessing
├── train.py                 # Training script
├── evaluate.py              # Evaluation script
├── predict.py              # Prediction and recommendation interface
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

---

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Required packages:
- `torch`: PyTorch framework
- `numpy`, `pandas`: Data handling
- `scikit-learn`: Metrics and preprocessing
- `matplotlib`: Visualization
- `tqdm`: Progress bars

### 2. Prepare Data

Your data should be in CSV format with the following columns:

#### ASSISTments Dataset (Secondary)
```csv
student_id,question_id,correct,timestamp
S001,1,1,2020-01-01 10:00:00
S001,5,0,2020-01-01 10:05:00
...
```

#### Primary Dataset (from your web system)
```csv
student_id,question_id,topic_id,is_correct,attempt_order,timestamp,response_time
S001,1,1,1,1,2025-03-01 10:00:00,45
S001,5,2,0,2,2025-03-01 10:05:00,120
...
```

Place datasets in the `data/` directory:
- `data/assistments_2009_2010.csv` (ASSISTments data)
- `data/primary_interactions.csv` (Your web system data)

---

## Configuration

Edit `config.py` to adjust model and training parameters:

```python
# Model Architecture
MODEL_CONFIG = {
    'hidden_size': 128,      # LSTM hidden dimension
    'num_layers': 2,         # Number of LSTM layers
    'dropout': 0.3,          # Dropout rate
}

# Training Parameters
TRAINING_CONFIG = {
    'batch_size': 32,
    'learning_rate': 0.001,
    'num_epochs': 50,
    'early_stopping_patience': 10,
    'device': 'cuda',        # 'cuda' for GPU, 'cpu' for CPU
}

# Data Processing
DATA_CONFIG = {
    'max_sequence_length': 100,
    'min_interactions': 5,   # Minimum interactions per student
}
```

---

## Training Pipeline

### Step 1: Train the DKT Model

Train on ASSISTments dataset (secondary data):

```bash
python train.py
```

**What happens:**
1. Loads ASSISTments dataset
2. Preprocesses and creates sequences
3. Splits into train/val/test (60/20/20)
4. Trains LSTM-based DKT model
5. Saves best checkpoint with early stopping
6. Outputs:
   - `checkpoints/dkt_model.pth` - Trained model
   - `results/training_history.json` - Loss and accuracy curves
   - `results/training_summary.json` - Training metadata
   - `results/preprocessor.pkl` - Data preprocessing info

**Output Example:**
```
============================================================
STEP 1: Loading ASSISTments Dataset
============================================================
Loading ASSISTments data from data/assistments_2009_2010.csv...
Loaded 500000 interactions from 4217 students

============================================================
STEP 2: Preprocessing Data
============================================================
Total unique questions: 189
Created 5234 sequences from 4217 students

============================================================
STEP 3: Building Model
============================================================
DKTModel(...)
Total parameters: 524,288
Trainable parameters: 524,288

============================================================
STEP 4: Training Model
============================================================
Epoch 1/50
Train Loss: 0.6231 | Train Acc: 0.6234
Val Loss: 0.5891 | Val Acc: 0.6512
...
```

---

### Step 2: Evaluate on Primary Dataset

Evaluate the trained model on your primary dataset:

```bash
python evaluate.py
```

**What happens:**
1. Loads trained DKT model
2. Loads primary dataset from web system
3. Evaluates on primary data
4. Computes metrics: Accuracy, Precision, Recall, AUC
5. Saves results

**Output Example:**
```
============================================================
EVALUATION RESULTS
============================================================
Accuracy:  0.7523
Precision: 0.7412
Recall:    0.8234
AUC:       0.8156

Confusion Matrix:
  True Negatives:  3421
  False Positives: 512
  False Negatives: 234
  True Positives:  2890
```

---

### Step 3: Generate Recommendations

Use the trained model to generate personalized recommendations:

```bash
python predict.py
```

**Code Example:**

```python
from predict import DKTPredictor

# Initialize predictor
predictor = DKTPredictor()

# Student's interaction history
student_interactions = [
    (1, 1),   # Question 1, Correct
    (5, 0),   # Question 5, Incorrect
    (2, 1),   # Question 2, Correct
    (3, 1),   # Question 3, Correct
]

# Predict mastery
mastery = predictor.predict_mastery(student_interactions)
print(f"Overall Mastery: {mastery['overall']:.4f}")
print(f"Trend: {mastery['trend']}")  # 'improving', 'declining', 'stable'

# Predict next question performance
next_perf = predictor.predict_next_performance(student_interactions, 7)
print(f"Predicted performance on Q7: {next_perf:.4f}")

# Generate recommendations
recommendations = predictor.generate_recommendations(
    student_interactions,
    {7: 'Variables', 8: 'Functions', 9: 'Loops'}
)

print(f"Mastery Level: {recommendations['mastery_level']}")
print("Recommended Questions:")
for rec in recommendations['recommended_questions']:
    print(f"  - Q{rec['question_id']}: {rec['predicted_performance']:.4f}")
```

---

## Model Architecture

### DKT Model Layers

```
Input: [(question_id, is_correct), ...]
  ↓
[Question Embedding] + [Correctness Embedding] + [Question Feature]
  ↓
[LSTM Layer 1] (128 hidden units)
  ↓
[LSTM Layer 2] (128 hidden units)
  ↓
[Dense Layer 1] (64 units, ReLU)
  ↓
[Dense Layer 2] (1 unit, Sigmoid)
  ↓
Output: [mastery_probability for next question]
```

### Input Format

**Question ID & Correctness Sequences:**
```python
# For a student with 10 interactions:
question_ids = torch.tensor([
    [1, 5, 2, 3, 4, 6, 7, 8, 9, 10]  # Question IDs
])

correctness = torch.tensor([
    [1, 0, 1, 1, 0, 1, 0, 1, 1, 1]   # Correctness (0 or 1)
])
```

### Output Interpretation

The model outputs **mastery probability** (0-1) for the next question:
- **0.8-1.0**: High mastery, student likely to answer correctly
- **0.5-0.8**: Intermediate mastery, mixed performance expected
- **0.2-0.5**: Low mastery, student needs more practice
- **0.0-0.2**: Very low mastery, student should focus on fundamentals

---

## Data Preprocessing

The `DataPreprocessor` class handles:

1. **Loading**: Reads CSV files with flexible column names
2. **Cleaning**: Removes invalid data and handles missing values
3. **Encoding**: Maps question IDs to sequential indices
4. **Sequencing**: Converts interactions into variable-length sequences
5. **Padding**: Pads sequences to fixed length for batch processing
6. **Splitting**: Divides data into train/val/test sets

### Example Usage

```python
from data_preprocessor import DataPreprocessor

preprocessor = DataPreprocessor()

# Load data
df = preprocessor.load_assistments_data('data/assistments_2009_2010.csv')

# Build question mapping
preprocessor.build_question_map(df)

# Create sequences
seqs_q, seqs_c, student_ids = preprocessor.create_sequences(
    df,
    max_length=100,
    min_interactions=5
)

# Prepare for training
X_q, X_c, lengths = preprocessor.prepare_training_data(
    seqs_q, seqs_c
)
```

---

## Training Details

### Loss Function
- **Binary Cross-Entropy (BCE)** with mastery prediction as target
- Custom `DKTLoss` class for weighted loss computation

### Optimization
- **Adam optimizer** with learning rate 0.001
- **Gradient clipping** (max_norm=1.0) to prevent exploding gradients
- **Early stopping** with patience=10 epochs

### Regularization
- Dropout (0.3) on embeddings and LSTM outputs
- L2 weight decay (1e-5)
- Batch normalization via LSTM dropout

### Learning Rate Schedule (Optional)
You can add learning rate scheduling:

```python
from torch.optim.lr_scheduler import ReduceLROnPlateau

scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
# In training loop:
scheduler.step(val_loss)
```

---

## Evaluation Metrics

### Accuracy
Proportion of correct predictions out of total.

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

### Precision
Of predicted positives, how many were actually positive.

$$\text{Precision} = \frac{TP}{TP + FP}$$

### Recall
Of actual positives, how many were predicted positive.

$$\text{Recall} = \frac{TP}{TP + FN}$$

### AUC (Area Under Curve)
Measures the model's ability to distinguish between classes across all thresholds.

---

## Troubleshooting

### Issue: "Out of Memory" Error
**Solution:** Reduce batch size in `config.py`:
```python
TRAINING_CONFIG['batch_size'] = 16  # from 32
```

### Issue: Model not improving
**Solution:** Try these adjustments:
```python
TRAINING_CONFIG['learning_rate'] = 0.0005  # Lower LR
MODEL_CONFIG['hidden_size'] = 64  # Smaller model
DATA_CONFIG['min_interactions'] = 10  # More training interactions
```

### Issue: Poor evaluation performance
**Possible causes:**
1. Domain mismatch between ASSISTments and primary data
2. Different question difficulty distribution
3. Different student populations
4. Insufficient primary data

**Solution:** Fine-tune the model on primary data:
```python
# In train.py, after pre-training:
# Load checkpoint and continue training on primary data
trainer.load_checkpoint(PATHS['model_checkpoint'])
trainer.train(primary_train_loader, primary_val_loader, num_epochs=20)
```

---

## Integration with Web System

To integrate with your existing web system:

### 1. Export Recommendations via API

```python
# In your web system's recommendation API:
from predict import DKTPredictor

predictor = DKTPredictor()

def get_recommendations(student_id):
    # Fetch student interactions from database
    interactions = db.get_student_interactions(student_id)
    
    # Generate recommendations
    recommendations = predictor.generate_recommendations(
        interactions,
        available_questions
    )
    
    return {
        'mastery_level': recommendations['mastery_level'],
        'recommended_questions': recommendations['recommended_questions']
    }
```

### 2. Periodic Model Updates

Schedule periodic retraining on accumulated data:

```bash
# Daily retraining script
python train.py  # Pre-train on ASSISTments
python evaluate.py  # Evaluate on primary data
```

### 3. A/B Testing

Compare recommendation quality:
- Control: Traditional recommendations
- Treatment: DKT-based recommendations
- Metric: Learning outcome improvement

---

## Advanced Features

### Fine-tuning on Primary Data

After pre-training on ASSISTments, fine-tune on your primary dataset:

```python
# In train.py, modify main():
# 1. Pre-train on ASSISTments
trainer.train(assistments_train_loader, assistments_val_loader)

# 2. Load trained model
trainer.load_checkpoint(PATHS['model_checkpoint'])

# 3. Fine-tune on primary data
trainer.train(primary_train_loader, primary_val_loader, num_epochs=10)

# 4. Save fine-tuned model
trainer.save_checkpoint('checkpoints/dkt_model_finetuned.pth')
```

### Custom Concept Tracking

Extend the model to track mastery per concept/topic:

```python
def predict_concept_mastery(self, student_interactions, concept_id):
    """Predict mastery for specific concept"""
    # Filter interactions for this concept
    concept_interactions = [
        (q, c) for q, c in student_interactions 
        if concept_map[q] == concept_id
    ]
    return self.predict_mastery(concept_interactions)
```

### Multi-objective Learning

Combine DKT loss with auxiliary tasks:

```python
# Predict both mastery AND learning time
class DKTMultiTask(DKTModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.time_head = nn.Linear(128, 1)
    
    def forward(self, question_ids, is_correct):
        mastery_output, hidden = super().forward(question_ids, is_correct)
        time_output = self.time_head(hidden)
        return mastery_output, time_output
```

---

## Thesis Citation

If you use this implementation in your thesis, please reference:

```bibtex
@inproceedings{piech2015deep,
  title={Deep knowledge tracing},
  author={Piech, Chris and Bassen, Jonathan and Huang, Jonathan and Ganguli, Surya and Sahami, Mehran and Guibas, Leonidas J and Savarese, Silvio},
  booktitle={Advances in neural information processing systems},
  pages={505--513},
  year={2015}
}

@article{vihavainen2012analyzing,
  title={Analyzing a large corpus of tutorial interactions: implications for intelligent systems and student modeling},
  author={Vihavainen, Arto and P{\"a}{\"a}kk{\"o}nen, Teemu and Aroyo, Lora and Titterton, Nick and Vuorikari, Riina},
  journal={arXiv preprint arXiv:1206.0087},
  year={2012}
}
```

---

## References

1. **DKT Original Paper**: [Piech et al., 2015](https://arxiv.org/abs/1506.05908)
2. **ASSISTments Dataset**: https://sites.google.com/site/assistmentsdata/home
3. **PyTorch Documentation**: https://pytorch.org/docs/stable/index.html
4. **LSTM Explained**: http://colah.github.io/posts/2015-08-Understanding-LSTMs/

---

## Support & Contribution

For issues or improvements:
1. Check existing documentation
2. Review error messages carefully
3. Verify data format matches expectations
4. Check `results/` directory for training logs

---

## License

This project is for academic research purposes.

**Last Updated**: March 2026
