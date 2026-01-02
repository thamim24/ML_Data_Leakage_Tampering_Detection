"""
Quick Start Script - Run Complete ML Data Leakage Detection Pipeline
"""
import sys
import os

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*70)
    print(text.center(70))
    print("="*70 + "\n")

def run_step(step_num, title, module_path, func_name=None):
    """Run a pipeline step"""
    print(f"\n{'─'*70}")
    print(f"STEP {step_num}: {title}")
    print(f"{'─'*70}\n")
    
    try:
        # Import and run the module
        if func_name:
            # Import specific function
            module_name = os.path.splitext(os.path.basename(module_path))[0]
            sys.path.insert(0, os.path.dirname(module_path))
            module = __import__(module_name)
            func = getattr(module, func_name)
            func()
        else:
            # Run module as script
            with open(module_path, 'r', encoding='utf-8') as f:
                code = f.read()
            exec(compile(code, module_path, 'exec'), {'__name__': '__main__'})
        
        print(f"\n✓ Step {step_num} completed successfully")
        return True
        
    except Exception as e:
        print(f"\n✗ Step {step_num} failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run complete pipeline"""
    print_header("ML DATA LEAKAGE DETECTION PROTOTYPE")
    print("Running complete detection pipeline...\n")
    
    # Check if we're in the right directory
    if not os.path.exists('requirements.txt'):
        print("ERROR: Please run this script from the project root directory")
        print("Current directory:", os.getcwd())
        return
    
    steps = [
        (1, "Generate Synthetic Data", "data/generate_data.py"),
        (2, "Preprocess Features", "notebooks/preprocessing.py"),
        (3, "Behavioral Anomaly Detection", "notebooks/anomaly_detection.py"),
        (4, "Document Classification", "notebooks/document_classification.py"),
        (5, "Integrity Verification", "notebooks/integrity_verification.py"),
        (6, "Risk Scoring & Fusion", "notebooks/risk_scoring.py"),
    ]
    
    results = []
    
    for step_num, title, module_path in steps:
        success = run_step(step_num, title, module_path)
        results.append((step_num, title, success))
        
        if not success:
            print(f"\nPipeline stopped at step {step_num}")
            break
    
    # Summary
    print_header("PIPELINE EXECUTION SUMMARY")
    
    for step_num, title, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {status} - Step {step_num}: {title}")
    
    successful_steps = sum(1 for _, _, success in results if success)
    total_steps = len(steps)
    
    print(f"\nCompleted: {successful_steps}/{total_steps} steps")
    
    if successful_steps == total_steps:
        print("\n🎉 Pipeline completed successfully!")
        print("\nNext steps:")
        print("  1. View results in the outputs/ directory")
        print("  2. Launch dashboard: streamlit run dashboard/app.py")
        print("  3. Open Jupyter notebook: jupyter notebook notebooks/prototype_main.ipynb")
        print("\nOptional XAI features:")
        print("  - Install SHAP: pip install shap")
        print("    Then run: python notebooks/explainability_shap.py")
        print("  - Install LIME: pip install lime")
        print("    Then run: python notebooks/explainability_lime.py")
    else:
        print("\n⚠️ Pipeline completed with errors. Check logs above.")
    
    print("\n" + "="*70 + "\n")

if __name__ == "__main__":
    main()