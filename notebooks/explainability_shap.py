"""
SHAP Explainability for Behavioral Anomaly Detection
Provides interpretable explanations for IsolationForest predictions
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP not installed. Install with: pip install shap")

class BehavioralExplainer:
    """Generate SHAP explanations for behavioral anomaly detection"""
    
    def __init__(self, model_path='models/isolation_forest.pkl',
                 features_path='data/log_features.csv'):
        """
        Initialize explainer
        
        Args:
            model_path: Path to saved model
            features_path: Path to feature data
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP is required. Install with: pip install shap")
        
        # Load model
        print("Loading model...")
        model_data = joblib.load(model_path)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        
        # Load features
        print("Loading features...")
        self.features_df = pd.read_csv(features_path)
        self.user_ids = self.features_df['user_id'].values
        self.X = self.features_df.drop(columns=['user_id'])
        self.X_scaled = self.scaler.transform(self.X)
        
        print(f"Loaded {len(self.X)} samples with {len(self.feature_names)} features")
        
        self.explainer = None
        self.shap_values = None
    
    def create_explainer(self, method='tree', background_samples=100):
        """
        Create SHAP explainer
        
        Args:
            method: 'tree' or 'kernel' (tree is faster for IsolationForest)
            background_samples: Number of background samples for KernelExplainer
        """
        print(f"\nCreating SHAP explainer (method: {method})...")
        
        try:
            if method == 'tree':
                # TreeExplainer works directly with tree-based models
                self.explainer = shap.TreeExplainer(self.model)
                print("TreeExplainer created successfully")
            else:
                # KernelExplainer is model-agnostic but slower
                background = shap.kmeans(self.X_scaled, background_samples)
                self.explainer = shap.KernelExplainer(
                    self.model.score_samples,
                    background
                )
                print(f"KernelExplainer created with {background_samples} background samples")
        except Exception as e:
            print(f"TreeExplainer failed: {e}")
            print("Falling back to KernelExplainer...")
            background = shap.kmeans(self.X_scaled, background_samples)
            self.explainer = shap.KernelExplainer(
                self.model.score_samples,
                background
            )
    
    def compute_shap_values(self, sample_size=None):
        """
        Compute SHAP values for all or subset of samples
        
        Args:
            sample_size: Number of samples to explain (None = all)
        """
        if self.explainer is None:
            self.create_explainer()
        
        if sample_size and sample_size < len(self.X_scaled):
            indices = np.random.choice(len(self.X_scaled), sample_size, replace=False)
            X_explain = self.X_scaled[indices]
            print(f"\nComputing SHAP values for {sample_size} samples...")
        else:
            X_explain = self.X_scaled
            print(f"\nComputing SHAP values for all {len(X_explain)} samples...")
        
        try:
            self.shap_values = self.explainer.shap_values(X_explain)
            print("SHAP values computed successfully")
        except Exception as e:
            print(f"Error computing SHAP values: {e}")
            print("Using approximate method...")
            self.shap_values = self.explainer.shap_values(
                X_explain,
                nsamples=100
            )
    
    def plot_summary(self, output_path='xai_outputs/shap_summary_behavior.png',
                     max_display=10):
        """
        Create SHAP summary plot showing global feature importance
        
        Args:
            output_path: Path to save plot
            max_display: Maximum features to display
        """
        if self.shap_values is None:
            self.compute_shap_values()
        
        print(f"\nGenerating SHAP summary plot...")
        
        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            self.shap_values,
            self.X,
            feature_names=self.feature_names,
            max_display=max_display,
            show=False
        )
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Summary plot saved to {output_path}")
    
    def plot_force_plot(self, user_id, output_path=None):
        """
        Create SHAP force plot for specific user
        
        Args:
            user_id: User ID to explain
            output_path: Path to save plot (auto-generated if None)
        """
        if self.shap_values is None:
            self.compute_shap_values()
        
        # Find user index
        user_idx = np.where(self.user_ids == user_id)[0]
        if len(user_idx) == 0:
            print(f"User {user_id} not found")
            return
        
        idx = user_idx[0]
        
        if output_path is None:
            output_path = f'xai_outputs/shap_force_user_{user_id}.png'
        
        print(f"\nGenerating force plot for {user_id}...")
        
        try:
            # Create force plot
            plt.figure(figsize=(12, 3))
            shap.force_plot(
                self.explainer.expected_value,
                self.shap_values[idx],
                self.X.iloc[idx],
                feature_names=self.feature_names,
                matplotlib=True,
                show=False
            )
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"Force plot saved to {output_path}")
        except Exception as e:
            print(f"Error creating force plot: {e}")
            print("Creating waterfall plot instead...")
            self.plot_waterfall(user_id, output_path)
    
    def plot_waterfall(self, user_id, output_path=None):
        """
        Create SHAP waterfall plot for specific user
        
        Args:
            user_id: User ID to explain
            output_path: Path to save plot
        """
        if self.shap_values is None:
            self.compute_shap_values()
        
        # Find user index
        user_idx = np.where(self.user_ids == user_id)[0]
        if len(user_idx) == 0:
            print(f"User {user_id} not found")
            return
        
        idx = user_idx[0]
        
        if output_path is None:
            output_path = f'xai_outputs/shap_waterfall_user_{user_id}.png'
        
        print(f"\nGenerating waterfall plot for {user_id}...")
        
        # Create explanation object
        explanation = shap.Explanation(
            values=self.shap_values[idx],
            base_values=self.explainer.expected_value,
            data=self.X.iloc[idx].values,
            feature_names=self.feature_names
        )
        
        plt.figure(figsize=(10, 6))
        shap.waterfall_plot(explanation, show=False)
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Waterfall plot saved to {output_path}")
    
    def explain_top_anomalies(self, n=5):
        """
        Generate explanations for top N most anomalous users
        
        Args:
            n: Number of top anomalies to explain
        """
        # Get anomaly scores
        scores = self.model.score_samples(self.X_scaled)
        top_indices = np.argsort(scores)[:n]
        
        print(f"\nGenerating explanations for top {n} anomalies...")
        
        for rank, idx in enumerate(top_indices, 1):
            user_id = self.user_ids[idx]
            score = scores[idx]
            
            print(f"\n{rank}. User {user_id} (score: {score:.3f})")
            
            # Generate force plot
            self.plot_force_plot(user_id)
            
            # Show top contributing features
            if self.shap_values is not None:
                feature_contributions = dict(zip(
                    self.feature_names,
                    self.shap_values[idx]
                ))
                sorted_features = sorted(
                    feature_contributions.items(),
                    key=lambda x: abs(x[1]),
                    reverse=True
                )[:5]
                
                print("  Top contributing features:")
                for feat, contrib in sorted_features:
                    direction = "increases" if contrib > 0 else "decreases"
                    print(f"    - {feat}: {direction} anomaly score (SHAP: {contrib:.3f})")


def generate_shap_explanations(model_path='models/isolation_forest.pkl',
                               features_path='data/log_features.csv',
                               output_dir='xai_outputs'):
    """
    Main function to generate SHAP explanations
    
    Args:
        model_path: Path to trained model
        features_path: Path to feature data
        output_dir: Directory for output files
    """
    if not SHAP_AVAILABLE:
        print("ERROR: SHAP is not installed.")
        print("Install with: pip install shap")
        print("\nSkipping SHAP explanations...")
        return None
    
    print("="*60)
    print("SHAP EXPLAINABILITY FOR BEHAVIORAL DETECTION")
    print("="*60)
    
    try:
        # Initialize explainer
        explainer = BehavioralExplainer(model_path, features_path)
        
        # Create explainer and compute SHAP values
        explainer.create_explainer(method='tree')
        explainer.compute_shap_values()
        
        # Generate summary plot
        print("\n" + "="*60)
        print("GENERATING VISUALIZATIONS")
        print("="*60)
        explainer.plot_summary()
        
        # Explain top anomalies
        explainer.explain_top_anomalies(n=5)
        
        print("\n" + "="*60)
        print("SHAP EXPLANATIONS COMPLETE")
        print("="*60)
        print(f"\nOutputs saved to {output_dir}/")
        
        return explainer
        
    except Exception as e:
        print(f"\nError generating SHAP explanations: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == '__main__':
    if SHAP_AVAILABLE:
        explainer = generate_shap_explanations()
    else:
        print("\nTo use SHAP explanations, install the package:")
        print("  pip install shap")
        print("\nNote: SHAP is optional and not required for the prototype to run.")