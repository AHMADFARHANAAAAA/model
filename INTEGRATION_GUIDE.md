"""
Integration Guide: DKT Model with Web System
Practical steps to integrate DKT predictions with your existing web application
"""

# ==============================================================================
# STEP 1: PREPARE YOUR DATA FOR TRAINING
# ==============================================================================

"""
Your web system stores interactions in the database. You need to export them
to CSV format for DKT training.

Database Schema (from prisma/schema.prisma):
- Interaction: student_id, question_id, selected_answer, is_correct, 
              attemptOrder, timestamp, timeSpent
- Question: id, topicId, skillId, content, difficulty, options, 
           correctAnswer, explanation
- Student: id, studentId, name, kelas (class)

SQL Query to Export:
"""

EXPORT_QUERY = """
SELECT 
    i.studentId as student_id,
    i.questionId as question_id,
    (SELECT topicId FROM questions WHERE id = i.questionId) as topic_id,
    i.isCorrect as is_correct,
    i.attemptOrder as attempt_order,
    i.timestamp,
    i.timeSpent as response_time
FROM interactions i
ORDER BY i.studentId, i.timestamp
"""

# Export to CSV:
# csv_file = pd.read_sql(EXPORT_QUERY, db_connection)
# csv_file.to_csv('data/primary_interactions.csv', index=False)


# ==============================================================================
# STEP 2: PYTHON BACKEND INTEGRATION
# ==============================================================================

"""
Create a Python service in your web system to handle DKT predictions.
This bridges your Node.js/Next.js backend with the Python ML model.
"""

# File: python/dkt_recommendation_service.py

from pathlib import Path
import sys
import json
import numpy as np
from typing import List, Dict, Tuple

# Add DKT model path
sys.path.insert(0, str(Path(__file__).parent.parent / 'DKT_Model'))

from dkt_model import DKTModel
from predict import DKTPredictor


class DKTRecommendationService:
    """
    Service for DKT-based recommendations
    Integrates between web system and ML model
    """
    
    def __init__(self):
        self.predictor = None
        self._initialize()
    
    def _initialize(self):
        """Initialize predictor on service startup"""
        try:
            self.predictor = DKTPredictor()
            print("✓ DKT Model initialized successfully")
        except Exception as e:
            print(f"✗ Failed to initialize DKT model: {e}")
            self.predictor = None
    
    def get_student_recommendations(
        self,
        student_id: str,
        interactions: List[Dict]
    ) -> Dict:
        """
        Get recommendations for a student
        
        Args:
            student_id: Student ID
            interactions: List of interaction dicts with:
                - question_id (int)
                - is_correct (0 or 1)
                - timestamp
        
        Returns:
            Recommendation dict for API response
        """
        if not self.predictor:
            return {'error': 'Model not initialized'}
        
        try:
            # Convert interactions
            interaction_tuples = [
                (int(i['question_id']), int(i['is_correct']))
                for i in interactions
            ]
            
            if len(interaction_tuples) < 5:
                return {
                    'status': 'insufficient_data',
                    'message': 'Need at least 5 interactions for recommendations'
                }
            
            # Generate recommendations
            available_questions = self._get_available_questions()
            recommendations = self.predictor.generate_recommendations(
                interaction_tuples,
                available_questions
            )
            
            return {
                'status': 'success',
                'student_id': student_id,
                'mastery_level': recommendations['mastery_level'],
                'overall_mastery': round(recommendations['overall_mastery'], 4),
                'mastery_trend': recommendations['mastery_trend'],
                'recommended_questions': [
                    {
                        'question_id': rec['question_id'],
                        'topic': rec['topic'],
                        'predicted_performance': round(rec['predicted_performance'], 4),
                        'difficulty': rec['difficulty']
                    }
                    for rec in recommendations['recommended_questions']
                ]
            }
        
        except ValueError as e:
            return {'error': str(e), 'status': 'error'}
        except Exception as e:
            return {'error': f'Unexpected error: {str(e)}', 'status': 'error'}
    
    def predict_next_question_performance(
        self,
        student_id: str,
        interactions: List[Dict],
        question_id: int
    ) -> Dict:
        """
        Predict student's performance on a specific question
        
        Returns probability of correctness (0-1)
        """
        if not self.predictor:
            return {'error': 'Model not initialized'}
        
        try:
            interaction_tuples = [
                (int(i['question_id']), int(i['is_correct']))
                for i in interactions
            ]
            
            performance = self.predictor.predict_next_performance(
                interaction_tuples,
                int(question_id)
            )
            
            return {
                'status': 'success',
                'student_id': student_id,
                'question_id': question_id,
                'predicted_performance': round(performance, 4),
                'difficulty': self._get_difficulty_label(performance)
            }
        
        except Exception as e:
            return {'error': str(e), 'status': 'error'}
    
    def batch_recommend(
        self,
        interactions_data: List[Dict]
    ) -> Dict:
        """
        Batch recommendations for multiple students
        
        Args:
            interactions_data: List of student interaction records
        
        Returns:
            Dict with recommendations per student
        """
        results = {}
        
        for student_id, interactions in interactions_data.items():
            results[student_id] = self.get_student_recommendations(
                student_id,
                interactions
            )
        
        return results
    
    @staticmethod
    def _get_available_questions() -> Dict[int, str]:
        """Get available questions from database"""
        # This should query your database for available questions
        # Example:
        # db = get_db_connection()
        # questions = db.execute("SELECT id, topic FROM questions")
        # return {q['id']: q['topic'] for q in questions}
        
        # Placeholder
        return {}
    
    @staticmethod
    def _get_difficulty_label(probability: float) -> str:
        """Convert probability to difficulty label"""
        if probability >= 0.7:
            return 'Easy'
        elif probability >= 0.5:
            return 'Medium'
        else:
            return 'Hard'


# ==============================================================================
# STEP 3: NODE.JS/NEXT.JS API ENDPOINT
# ==============================================================================

"""
Create an API route in your Next.js app to handle recommendation requests.

File: app/api/recommendation/[studentId]/route.ts
"""

NEXT_JS_ENDPOINT = """
import { type NextRequest, NextResponse } from 'next/server';
import { PrismaClient } from '@prisma/client';

const prisma = new PrismaClient();

// Call to Python service (can be direct or via HTTP)
async function getDKTRecommendations(studentId: string) {
  // Option 1: Direct Python call (requires python subprocess)
  // const { spawn } = require('child_process');
  // const python = spawn('python', ['python/dkt_service.py', studentId]);
  
  // Option 2: HTTP call to Python service
  const response = await fetch('http://localhost:8000/recommend', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ studentId })
  });
  
  return response.json();
}

export async function GET(
  request: NextRequest,
  { params }: { params: { studentId: string } }
) {
  const { studentId } = params;
  
  try {
    // Get student interactions
    const interactions = await prisma.interaction.findMany({
      where: { studentId },
      select: {
        questionId: true,
        isCorrect: true,
        timestamp: true
      },
      orderBy: { timestamp: 'asc' }
    });
    
    if (interactions.length === 0) {
      return NextResponse.json({
        status: 'no_data',
        message: 'No interactions found for student'
      }, { status: 404 });
    }
    
    // Get DKT recommendations
    const recommendations = await getDKTRecommendations(studentId);
    
    return NextResponse.json(recommendations);
  
  } catch (error) {
    console.error('Recommendation error:', error);
    return NextResponse.json(
      { error: 'Failed to generate recommendations' },
      { status: 500 }
    );
  }
}
"""

# Or use a simpler approach with Flask backend:

FLASK_SERVICE = """
# python/dkt_service.py
from flask import Flask, jsonify, request
from dkt_recommendation_service import DKTRecommendationService
import json

app = Flask(__name__)
service = DKTRecommendationService()

@app.route('/recommend', methods=['POST'])
def get_recommendations():
    data = request.json
    student_id = data.get('student_id')
    interactions = data.get('interactions')
    
    recommendations = service.get_student_recommendations(
        student_id,
        interactions
    )
    
    return jsonify(recommendations)

@app.route('/predict', methods=['POST'])
def predict_performance():
    data = request.json
    result = service.predict_next_question_performance(
        data.get('student_id'),
        data.get('interactions'),
        data.get('question_id')
    )
    
    return jsonify(result)

if __name__ == '__main__':
    print("Starting DKT recommendation service...")
    app.run(host='localhost', port=8000, debug=False)
"""


# ==============================================================================
# STEP 4: INTEGRATION WITH LEARNING PAGE
# ==============================================================================

"""
File: app/learn/page.tsx

Update the learning page to display recommendations
"""

LEARNING_PAGE_INTEGRATION = """
'use client';

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';

interface Recommendation {
  question_id: number;
  topic: string;
  predicted_performance: number;
  difficulty: string;
}

interface RecommendationData {
  status: string;
  mastery_level: string;
  overall_mastery: number;
  recommended_questions: Recommendation[];
}

export default function LearnPage() {
  const [student, setStudent] = useState<any>(null);
  const [recommendations, setRecommendations] = useState<RecommendationData | null>(null);
  const [loading, setLoading] = useState(true);
  const router = useRouter();

  useEffect(() => {
    // Get student info
    const studentData = sessionStorage.getItem('student');
    if (!studentData) {
      router.push('/');
      return;
    }
    
    const student = JSON.parse(studentData);
    setStudent(student);
    
    // Fetch recommendations
    const fetchRecommendations = async () => {
      try {
        const res = await fetch(`/api/recommendation/${student.id}`, {
          method: 'GET',
        });
        
        const data = await res.json();
        setRecommendations(data);
      } catch (error) {
        console.error('Failed to fetch recommendations:', error);
      } finally {
        setLoading(false);
      }
    };
    
    fetchRecommendations();
  }, [router]);

  if (loading) return <div>Loading recommendations...</div>;

  return (
    <div className="p-8">
      <h1 className="text-3xl font-bold mb-6">Learning Path</h1>
      
      {recommendations && (
        <div className="bg-blue-50 p-6 rounded-lg mb-8">
          <h2 className="text-xl font-semibold mb-4">Your Learning Status</h2>
          
          <div className="grid grid-cols-2 gap-4 mb-6">
            <div className="bg-white p-4 rounded">
              <p className="text-gray-600">Mastery Level</p>
              <p className="text-2xl font-bold">{recommendations.mastery_level}</p>
            </div>
            <div className="bg-white p-4 rounded">
              <p className="text-gray-600">Overall Progress</p>
              <p className="text-2xl font-bold">
                {(recommendations.overall_mastery * 100).toFixed(1)}%
              </p>
            </div>
          </div>
          
          <h3 className="font-semibold mb-4">Recommended Questions</h3>
          <div className="space-y-3">
            {recommendations.recommended_questions.map((rec) => (
              <button
                key={rec.question_id}
                className="w-full bg-white p-4 rounded hover:bg-blue-100 transition text-left"
              >
                <div className="flex justify-between items-center">
                  <div>
                    <p className="font-semibold">Question {rec.question_id}</p>
                    <p className="text-sm text-gray-600">{rec.topic}</p>
                  </div>
                  <span className={`px-3 py-1 rounded text-sm font-medium \${
                    rec.difficulty === 'Easy' ? 'bg-green-100 text-green-800' :
                    rec.difficulty === 'Medium' ? 'bg-yellow-100 text-yellow-800' :
                    'bg-red-100 text-red-800'
                  }`}>
                    {rec.difficulty}
                  </span>
                </div>
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
"""


# ==============================================================================
# STEP 5: PERIODIC RETRAINING SCHEDULER
# ==============================================================================

"""
Set up periodic retraining to keep the model updated with new data.

Create a cron job or scheduled task:
"""

RETRAINING_SCRIPT = """
# scripts/retrain_dkt_daily.py

import subprocess
import schedule
import time
from datetime import datetime
from pathlib import Path
import logging

logging.basicConfig(
    filename='logs/dkt_retraining.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def export_data():
    \"\"\"Export interactions from database to CSV\"\"\"
    logging.info("Exporting interaction data...")
    # Execute SQL query and save to CSV
    subprocess.run(['python', 'scripts/export_interactions.py'])

def train_model():
    \"\"\"Retrain DKT model\"\"\"
    logging.info("Starting DKT model retraining...")
    result = subprocess.run(['python', 'DKT_Model/train.py'])
    if result.returncode == 0:
        logging.info("Model retraining completed successfully")
    else:
        logging.error("Model retraining failed")

def evaluate_model():
    \"\"\"Evaluate retrained model\"\"\"
    logging.info("Evaluating retrained model...")
    subprocess.run(['python', 'DKT_Model/evaluate.py'])

def scheduled_retraining():
    \"\"\"Daily retraining task\"\"\"
    logging.info("="*50)
    logging.info(f"Starting scheduled retraining at {datetime.now()}")
    
    try:
        export_data()
        train_model()
        evaluate_model()
        logging.info("Scheduled retraining completed successfully")
    except Exception as e:
        logging.error(f"Scheduled retraining failed: {e}")

# Schedule daily retraining at 2 AM
schedule.every().day.at("02:00").do(scheduled_retraining)

if __name__ == '__main__':
    logging.info("DKT Retraining Scheduler Started")
    
    while True:
        schedule.run_pending()
        time.sleep(60)
"""


# ==============================================================================
# STEP 6: MONITORING AND VALIDATION
# ==============================================================================

"""
Monitor model performance and validate predictions
"""

MONITORING_SCRIPT = """
# scripts/monitor_dkt_performance.py

import json
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

class DKTPerformanceMonitor:
    def __init__(self):
        self.results_dir = Path('DKT_Model/results')
        self.log_file = Path('logs/dkt_performance.json')
    
    def log_metrics(self):
        \"\"\"Log current model metrics\"\"\"
        try:
            with open(self.results_dir / 'evaluation_metrics.json', 'r') as f:
                metrics = json.load(f)
            
            # Append to log
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'accuracy': metrics.get('accuracy'),
                'auc': metrics.get('auc'),
                'precision': metrics.get('precision'),
                'recall': metrics.get('recall')
            }
            
            logs = []
            if self.log_file.exists():
                with open(self.log_file, 'r') as f:
                    logs = json.load(f)
            
            logs.append(log_entry)
            
            with open(self.log_file, 'w') as f:
                json.dump(logs, f, indent=2)
            
            print("✓ Metrics logged")
        
        except Exception as e:
            print(f"✗ Failed to log metrics: {e}")
    
    def detect_performance_degradation(self, threshold=0.05):
        \"\"\"Alert if model performance degrades\"\"\"
        if not self.log_file.exists():
            return False
        
        with open(self.log_file, 'r') as f:
            logs = json.load(f)
        
        if len(logs) < 2:
            return False
        
        recent = logs[-1]
        previous = logs[-2]
        
        accuracy_drop = previous['accuracy'] - recent['accuracy']
        
        if accuracy_drop > threshold:
            print(f"⚠ Warning: Accuracy dropped by {accuracy_drop:.2%}")
            print(f"  Previous: {previous['accuracy']:.2%}")
            print(f"  Current:  {recent['accuracy']:.2%}")
            return True
        
        return False

if __name__ == '__main__':
    monitor = DKTPerformanceMonitor()
    monitor.log_metrics()
    monitor.detect_performance_degradation()
"""


# ==============================================================================
# SUMMARY: INTEGRATION CHECKLIST
# ==============================================================================

INTEGRATION_CHECKLIST = """
INTEGRATION CHECKLIST:

Pre-Integration:
  ☐ Train DKT model on ASSISTments dataset (python DKT_Model/train.py)
  ☐ Evaluate on primary dataset (python DKT_Model/evaluate.py)
  ☐ Verify evaluation metrics are acceptable (>70% accuracy)

Backend Integration:
  ☐ Create python/dkt_recommendation_service.py
  ☐ Install required Python dependencies (pip install -r requirements.txt)
  ☐ Test Python service independently
  ☐ Create Flask/FastAPI endpoint for recommendations
  ☐ Add /api/recommendation/{studentId} endpoint to Next.js

Frontend Integration:
  ☐ Update learn/page.tsx to display recommendations
  ☐ Add recommendation cards to student dashboard
  ☐ Show mastery level and learning trend
  ☐ Display recommended questions with difficulty

Data Pipeline:
  ☐ Create scripts/export_interactions.py (exports DB to CSV)
  ☐ Test data export (verify CSV format)
  ☐ Create retraining scheduler
  ☐ Set up cron jobs for daily retraining

Monitoring:
  ☐ Create monitoring script
  ☐ Set up performance logging
  ☐ Configure alerts for model degradation
  ☐ Set up logs directory with proper permissions

Testing:
  ☐ Test with sample student interactions
  ☐ Verify recommendation quality
  ☐ Load test with multiple concurrent requests
  ☐ Test edge cases (new students, many interactions, etc.)

Deployment:
  ☐ Configure production paths in config.py
  ☐ Set up GPU/CPU resource allocation
  ☐ Create deployment documentation
  ☐ Test in staging environment
  ☐ Monitor first week of production carefully

Documentation:
  ☐ Document API endpoints
  ☐ Create troubleshooting guide
  ☐ Document retraining schedule
  ☐ Create runbook for model updates
"""

print(__doc__)
