"""
Data Export Script
Export student interaction data from Prisma database to CSV for DKT training
"""

import csv
import sys
from pathlib import Path
from datetime import datetime

# This script needs to be adapted to your database setup
# Below is a template using Prisma (from your schema.prisma)


def export_interactions_to_csv(output_path: str = "data/primary_interactions.csv"):
    """
    Export interactions from database to CSV
    
    This template uses Prisma Python client or direct database queries
    """
    
    print("Starting data export...")
    print(f"Output file: {output_path}")
    
    # Create output directory if it doesn't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Option 1: Using Prisma with subprocess
    # (Requires running through Node.js since Prisma is primarily Node-based)
    
    print("\nUsing Node.js to query database through Prisma...")
    
    try:
        import subprocess
        
        # Create a temporary Node.js script to export data
        export_script = '''
const { PrismaClient } = require('@prisma/client');
const fs = require('fs');
const path = require('path');
const { createObjectCsvWriter } = require('csv-writer');

const prisma = new PrismaClient();

async function exportInteractions() {
    console.log('Fetching interactions from database...');
    
    const interactions = await prisma.interaction.findMany({
        include: {
            question: true,
            student: true
        },
        orderBy: [
            { studentId: 'asc' },
            { timestamp: 'asc' }
        ]
    });
    
    console.log(`Found ${interactions.length} interactions`);
    
    // Convert to CSV format
    const records = interactions.map(i => ({
        student_id: i.studentId,
        question_id: i.questionId,
        topic_id: i.question?.topicId || 1,
        is_correct: i.isCorrect ? 1 : 0,
        attempt_order: i.attemptOrder,
        timestamp: i.timestamp.toISOString(),
        response_time: i.timeSpent || 0
    }));
    
    // Write CSV
    const csvWriter = createObjectCsvWriter({
        path: 'data/primary_interactions.csv',
        header: [
            { id: 'student_id', title: 'student_id' },
            { id: 'question_id', title: 'question_id' },
            { id: 'topic_id', title: 'topic_id' },
            { id: 'is_correct', title: 'is_correct' },
            { id: 'attempt_order', title: 'attempt_order' },
            { id: 'timestamp', title: 'timestamp' },
            { id: 'response_time', title: 'response_time' }
        ]
    });
    
    await csvWriter.writeRecords(records);
    console.log('✓ Interactions exported to CSV');
    
    await prisma.$disconnect();
}

exportInteractions()
    .catch(error => {
        console.error('Export failed:', error);
        process.exit(1);
    });
        '''
        
        # Write temporary export script
        temp_script = Path('temp_export.js')
        temp_script.write_text(export_script)
        
        # Run the Node.js script
        result = subprocess.run(
            ['node', str(temp_script)],
            cwd='Code/Main Code/soal',
            capture_output=True,
            text=True
        )
        
        print(result.stdout)
        if result.returncode != 0:
            print(f"Error: {result.stderr}")
        
        # Clean up
        temp_script.unlink()
        
    except Exception as e:
        print(f"Failed to export via Node.js: {e}")
        print("\nAlternative: Use direct SQL query")
        print("=" * 60)
        print("""
# Option 2: Direct SQL Export

If using SQLite (as in your schema), use:

sqlite3 dev.db << EOF
.headers on
.mode csv
.output data/primary_interactions.csv
SELECT 
    interactions.studentId as student_id,
    interactions.questionId as question_id,
    questions.topicId as topic_id,
    CASE WHEN interactions.isCorrect THEN 1 ELSE 0 END as is_correct,
    interactions.attemptOrder as attempt_order,
    interactions.timestamp,
    interactions.timeSpent as response_time
FROM interactions
JOIN questions ON questions.id = interactions.questionId
ORDER BY interactions.studentId, interactions.timestamp;
EOF

# If using PostgreSQL, use:
psql your_database << EOF
\\COPY (
    SELECT 
        studentId as student_id,
        questionId as question_id,
        topicId as topic_id,
        isCorrect::int as is_correct,
        attemptOrder as attempt_order,
        timestamp,
        timeSpent as response_time
    FROM interactions
    ORDER BY studentId, timestamp
) TO 'data/primary_interactions.csv' WITH CSV HEADER;
EOF
        """)
        return False
    
    return True


def export_assistments_data(
    source_path: str = "../../Dataset/assistments_2009_2010.csv",
    output_path: str = "data/assistments_2009_2010.csv"
) -> bool:
    """
    Copy ASSISTments dataset to data directory
    
    The ASSISTments dataset should have columns:
    - student_id (or 'user_id')
    - question_id (or 'problem_id')
    - correct (or 'outcome')
    - timestamp (or use default)
    """
    
    print("\nSetting up ASSISTments dataset...")
    print(f"Source: {source_path}")
    print(f"Output: {output_path}")
    
    source_file = Path(source_path)
    output_file = Path(output_path)
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        if source_file.exists():
            # Copy file
            import shutil
            shutil.copy(source_file, output_file)
            print(f"✓ ASSISTments dataset copied")
            return True
        else:
            print(f"✗ ASSISTments dataset not found at {source_path}")
            print("\nTo download ASSISTments dataset:")
            print("  1. Visit: https://sites.google.com/site/assistmentsdata/home")
            print("  2. Register and download the dataset")
            print(f"  3. Place it at: {source_path}")
            return False
    
    except Exception as e:
        print(f"Error copying file: {e}")
        return False


def verify_data_structure(csv_path: str) -> bool:
    """
    Verify CSV structure matches DKT requirements
    """
    
    print(f"\nVerifying data structure: {csv_path}")
    
    try:
        import pandas as pd
        
        df = pd.read_csv(csv_path, nrows=100)
        
        required_cols = ['student_id', 'question_id', 'is_correct']
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"✗ Missing columns: {missing_cols}")
            print(f"  Found columns: {list(df.columns)}")
            return False
        
        print(f"✓ CSV structure is valid")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Rows: {len(df)}")
        print(f"  Unique students: {df['student_id'].nunique()}")
        print(f"  Unique questions: {df['question_id'].nunique()}")
        print(f"  Success rate: {df['is_correct'].mean():.2%}")
        
        return True
    
    except Exception as e:
        print(f"✗ Verification failed: {e}")
        return False


def main():
    """Main export routine"""
    
    print("=" * 70)
    print("DKT Dataset Export Tool")
    print("=" * 70)
    
    # Export primary data
    print("\n1. Exporting primary dataset from web system...")
    print("-" * 70)
    
    success_primary = export_interactions_to_csv()
    
    # Setup ASSISTments data
    print("\n2. Setting up ASSISTments dataset...")
    print("-" * 70)
    
    success_assistments = export_assistments_data()
    
    # Verify both datasets
    print("\n3. Verifying datasets...")
    print("-" * 70)
    
    if success_primary:
        verify_data_structure("data/primary_interactions.csv")
    
    if success_assistments:
        verify_data_structure("data/assistments_2009_2010.csv")
    
    # Summary
    print("\n" + "=" * 70)
    print("Export Summary")
    print("=" * 70)
    
    if success_primary and success_assistments:
        print("✓ All datasets ready for training!")
        print("\nNext steps:")
        print("  1. Run: python DKT_Model/train.py")
        print("  2. Run: python DKT_Model/evaluate.py")
        print("  3. Review: DKT_Model/results/evaluation_metrics.json")
    else:
        print("⚠ Some datasets are missing")
        print("  Complete the steps above before training")
    
    print()


if __name__ == '__main__':
    main()
