"""
Behavioral Anomaly Detection using IsolationForest
Detects unusual user behavior patterns that may indicate data leakage
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
import os
import json

class BehavioralAnomalyDetector:
    """Detect anomalous user behavior using IsolationForest"""
    
    def __init__(self, contamination=0.1, n_estimators=100, random_state=42):
        """
        Args:
            contamination: Expected proportion of anomalies in dataset
            n_estimators: Number of trees in the forest
            random_state: Random seed for reproducibility
        """
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def train(self, X, feature_names=None):
        """
        Train the anomaly detection model
        
        Args:
            X: Feature matrix (DataFrame or array)
            feature_names: List of feature names (optional)
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X_scaled = self.scaler.fit_transform(X)
        else:
            self.feature_names = feature_names
            X_scaled = self.scaler.fit_transform(X)
        
        # Train IsolationForest
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        self.model.fit(X_scaled)
        
        print(f"Trained IsolationForest with {self.n_estimators} estimators")
        print(f"Expected contamination rate: {self.contamination}")
        
    def predict(self, X):
        """
        Predict anomalies
        
        Args:
            X: Feature matrix
            
        Returns:
            predictions: -1 for anomalies, 1 for normal
            scores: Anomaly scores (lower = more anomalous)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        scores = self.model.score_samples(X_scaled)
        
        return predictions, scores
    
    def detect_anomalies(self, X, threshold_percentile=10):
        """
        Detect anomalies with detailed results
        
        Args:
            X: Feature matrix (DataFrame preferred)
            threshold_percentile: Percentile for anomaly threshold
            
        Returns:
            DataFrame with anomaly flags and scores
        """
        predictions, scores = self.predict(X)
        
        # Convert to DataFrame if needed
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names)
        
        results = X.copy()
        results['anomaly_score'] = scores
        results['is_anomaly'] = (predictions == -1).astype(int)
        
        # Add risk level based on score percentiles
        score_percentiles = np.percentile(scores, [10, 25, 50, 75, 90])
        
        def get_risk_level(score):
            if score < score_percentiles[0]:
                return 'critical'
            elif score < score_percentiles[1]:
                return 'high'
            elif score < score_percentiles[2]:
                return 'medium'
            else:
                return 'low'
        
        results['risk_level'] = results['anomaly_score'].apply(get_risk_level)
        
        return results
    
    def get_feature_importance_proxy(self, X):
        """
        Estimate feature importance using variance of scores when features are permuted
        This is a proxy for feature importance specific to anomaly detection
        
        Args:
            X: Feature matrix (DataFrame)
            
        Returns:
            DataFrame with feature importance scores
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names)
        
        baseline_scores = self.model.score_samples(self.scaler.transform(X))
        baseline_variance = np.var(baseline_scores)
        
        importances = {}
        
        for col in X.columns:
            X_permuted = X.copy()
            X_permuted[col] = np.random.permutation(X_permuted[col].values)
            
            permuted_scores = self.model.score_samples(self.scaler.transform(X_permuted))
            permuted_variance = np.var(permuted_scores)
            
            # Importance = change in variance
            importances[col] = abs(baseline_variance - permuted_variance)
        
        importance_df = pd.DataFrame({
            'feature': list(importances.keys()),
            'importance': list(importances.values())
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def save(self, path='models/isolation_forest.pkl'):
        """Save model and scaler"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'contamination': self.contamination,
            'n_estimators': self.n_estimators
        }
        joblib.dump(model_data, path)
        print(f"Model saved to {path}")
    
    @staticmethod
    def load(path='models/isolation_forest.pkl'):
        """Load model and scaler"""
        model_data = joblib.load(path)
        
        detector = BehavioralAnomalyDetector(
            contamination=model_data['contamination'],
            n_estimators=model_data['n_estimators']
        )
        detector.model = model_data['model']
        detector.scaler = model_data['scaler']
        detector.feature_names = model_data['feature_names']
        
        print(f"Model loaded from {path}")
        return detector


def train_and_evaluate(features_path='data/log_features.csv', 
                       output_path='outputs/anomaly_results.csv'):
    """
    Main function to train and evaluate anomaly detection model
    """
    print("="*60)
    print("BEHAVIORAL ANOMALY DETECTION")
    print("="*60)
    
    # Load features
    print("\n1. Loading features...")
    df = pd.read_csv(features_path)
    print(f"Loaded {len(df)} user records with {len(df.columns)-1} features")
    
    # Separate user_id from features
    user_ids = df['user_id'].values
    X = df.drop(columns=['user_id'])
    
    print(f"\nFeatures used: {X.columns.tolist()}")
    
    # Train model
    print("\n2. Training IsolationForest model...")
    detector = BehavioralAnomalyDetector(
        contamination=0.1,  # expect 10% anomalies
        n_estimators=100,
        random_state=42
    )
    detector.train(X)
    
    # Detect anomalies
    print("\n3. Detecting anomalies...")
    results = detector.detect_anomalies(X)
    results['user_id'] = user_ids
    
    # Reorder columns
    cols = ['user_id'] + [col for col in results.columns if col != 'user_id']
    results = results[cols]
    
    # Statistics
    n_anomalies = results['is_anomaly'].sum()
    print(f"\nDetected {n_anomalies} anomalous users ({n_anomalies/len(results)*100:.1f}%)")
    
    print("\nRisk level distribution:")
    print(results['risk_level'].value_counts())
    
    print("\nTop 10 most anomalous users:")
    top_anomalies = results.nsmallest(10, 'anomaly_score')[
        ['user_id', 'anomaly_score', 'risk_level', 'is_anomaly']
    ]
    print(top_anomalies.to_string(index=False))
    
    # Feature importance
    print("\n4. Computing feature importance...")
    importance_df = detector.get_feature_importance_proxy(X)
    print("\nFeature importance (proxy):")
    print(importance_df.to_string(index=False))
    
    # Save results
    print(f"\n5. Saving results to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results.to_csv(output_path, index=False)
    
    # Save feature importance
    importance_path = output_path.replace('anomaly_results.csv', 'feature_importance.csv')
    importance_df.to_csv(importance_path, index=False)
    
    # Save model
    detector.save()
    
    # Save summary statistics
    summary = {
        'total_users': len(results),
        'anomalous_users': int(n_anomalies),
        'anomaly_rate': float(n_anomalies/len(results)),
        'risk_levels': results['risk_level'].value_counts().to_dict(),
        'score_stats': {
            'mean': float(results['anomaly_score'].mean()),
            'std': float(results['anomaly_score'].std()),
            'min': float(results['anomaly_score'].min()),
            'max': float(results['anomaly_score'].max())
        }
    }
    
    summary_path = output_path.replace('anomaly_results.csv', 'anomaly_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "="*60)
    print("ANOMALY DETECTION COMPLETE")
    print("="*60)
    
    return results, detector


if __name__ == '__main__':
    results, detector = train_and_evaluate()