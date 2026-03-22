"""
Quick Start Script for DKT Model
Run this to test the complete pipeline without command line
"""

import sys
import os
from pathlib import Path

def setup_environment():
    """Setup Python path and check dependencies"""
    print("Checking dependencies...")
    
    try:
        import torch
        import numpy
        import pandas
        import sklearn
        print(f"✓ PyTorch {torch.__version__}")
        print(f"✓ NumPy {numpy.__version__}")
        print(f"✓ Pandas {pandas.__version__}")
        print(f"✓ Scikit-learn {sklearn.__version__}")
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print("\nInstall dependencies with:")
        print("  pip install -r requirements.txt")
        return False
    
    return True


def check_data_files():
    """Check if data files exist"""
    print("\nChecking data files...")
    
    data_dir = Path("data")
    assistments_file = data_dir / "assistments_2009_2010.csv"
    primary_file = data_dir / "primary_interactions.csv"
    
    if not data_dir.exists():
        print(f"Create directory: {data_dir}")
        data_dir.mkdir(parents=True, exist_ok=True)
    
    if not assistments_file.exists():
        print(f"✗ Missing: {assistments_file}")
        print("  Download from: https://sites.google.com/site/assistmentsdata/home")
        return False
    else:
        print(f"✓ Found: {assistments_file}")
    
    if not primary_file.exists():
        print(f"⚠ Optional: {primary_file}")
        print("  Will be needed for evaluation step")
    else:
        print(f"✓ Found: {primary_file}")
    
    return True


def run_training():
    """Run training pipeline"""
    print("\n" + "="*60)
    print("TRAINING DKT MODEL")
    print("="*60)
    
    try:
        from train import main as train_main
        train_main()
        return True
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_evaluation():
    """Run evaluation pipeline"""
    print("\n" + "="*60)
    print("EVALUATING DKT MODEL")
    print("="*60)
    
    # Check if primary data exists
    primary_file = Path("data/primary_interactions.csv")
    if not primary_file.exists():
        print("⚠ Skipping evaluation (primary data not found)")
        print(f"  Place your data at: {primary_file}")
        return False
    
    try:
        from evaluate import main as evaluate_main
        evaluate_main()
        return True
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_prediction_example():
    """Run prediction example"""
    print("\n" + "="*60)
    print("GENERATING RECOMMENDATIONS")
    print("="*60)
    
    try:
        from predict import example_usage
        example_usage()
        return True
    except Exception as e:
        print(f"Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        return False


def print_menu():
    """Print interactive menu"""
    print("\n" + "="*60)
    print("DKT MODEL - QUICK START")
    print("="*60)
    print("\nOptions:")
    print("  1. Check setup (dependencies and data)")
    print("  2. Train model on ASSISTments data")
    print("  3. Evaluate model on primary data")
    print("  4. Generate recommendations (example)")
    print("  5. Run complete pipeline (1 → 2 → 3 → 4)")
    print("  6. Exit")
    print("\nEnter option (1-6): ", end="")


def main():
    """Main entry point"""
    print("\n" + "="*60)
    print("Deep Knowledge Tracing - Quick Start Guide")
    print("="*60)
    
    # Non-interactive mode: run complete pipeline
    if len(sys.argv) > 1 and sys.argv[1] == '--auto':
        print("Running automatic pipeline...")
        
        if not setup_environment():
            return
        
        if not check_data_files():
            return
        
        if not run_training():
            print("Training failed. Check errors above.")
            return
        
        if not run_evaluation():
            print("Evaluation failed (optional).")
        
        if not run_prediction_example():
            print("Prediction failed.")
        
        print("\n" + "="*60)
        print("Pipeline Complete!")
        print("="*60)
        print("\nNext steps:")
        print("  1. Check results/training_history.json for training curves")
        print("  2. Check results/evaluation_metrics.json for test metrics")
        print("  3. Review predict.py for integration with your system")
        return
    
    # Interactive mode
    while True:
        print_menu()
        
        try:
            choice = input().strip()
            
            if choice == '1':
                if setup_environment() and check_data_files():
                    print("✓ Setup complete!")
            
            elif choice == '2':
                if not setup_environment():
                    continue
                run_training()
            
            elif choice == '3':
                if not setup_environment():
                    continue
                run_evaluation()
            
            elif choice == '4':
                if not setup_environment():
                    continue
                run_prediction_example()
            
            elif choice == '5':
                if not setup_environment():
                    continue
                if not check_data_files():
                    continue
                
                print("\nRunning complete pipeline...")
                if run_training():
                    if run_evaluation():
                        run_prediction_example()
                    else:
                        print("Evaluation skipped")
                
                print("\n✓ Pipeline complete!")
            
            elif choice == '6':
                print("\nGoodbye!")
                break
            
            else:
                print("Invalid option. Please enter 1-6.")
        
        except KeyboardInterrupt:
            print("\n\nInterrupted by user.")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == '__main__':
    main()
