"""
Risk Scoring and Fusion Module
Combines behavioral anomaly, document classification, and integrity signals
into unified risk scores and alerts
"""
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime

class RiskScorer:
    """Fuse multiple detection signals into unified risk scores"""
    
    def __init__(self, weights=None):
        """
        Initialize risk scorer with fusion weights
        
        Args:
            weights: Dictionary of weights for each component
                    Default: {'behavior': 0.4, 'classification': 0.3, 'integrity': 0.3}
        """
        if weights is None:
            self.weights = {
                'behavior': 0.4,
                'classification': 0.3,
                'integrity': 0.3
            }
        else:
            self.weights = weights
        
        # Normalize weights to sum to 1
        total = sum(self.weights.values())
        self.weights = {k: v/total for k, v in self.weights.items()}
        
        print(f"Risk scorer initialized with weights: {self.weights}")
    
    def normalize_anomaly_score(self, score):
        """
        Normalize anomaly score to 0-1 range (higher = more risky)
        IsolationForest scores are negative, more negative = more anomalous
        
        Args:
            score: Raw anomaly score
            
        Returns:
            Normalized score (0-1)
        """
        # IsolationForest scores typically range from -0.5 to 0.5
        # More negative = more anomalous
        normalized = (-score + 0.5) / 1.0  # Map to 0-1
        return np.clip(normalized, 0, 1)
    
    def get_classification_risk(self, sensitivity, confidence):
        """
        Convert classification to risk score
        
        Args:
            sensitivity: Sensitivity level (public/internal/confidential)
            confidence: Classification confidence
            
        Returns:
            Risk score (0-1)
        """
        sensitivity_scores = {
            'public': 0.1,
            'internal': 0.5,
            'confidential': 0.9
        }
        
        base_score = sensitivity_scores.get(sensitivity.lower(), 0.5)
        # Weight by confidence
        risk_score = base_score * confidence
        
        return risk_score
    
    def get_integrity_risk(self, tamper_detected, tamper_severity):
        """
        Convert integrity check to risk score
        
        Args:
            tamper_detected: Boolean, was tampering detected
            tamper_severity: Severity level (none/minor/moderate/major/unknown)
            
        Returns:
            Risk score (0-1)
        """
        if not tamper_detected:
            return 0.0
        
        severity_scores = {
            'none': 0.0,
            'minor': 0.3,
            'moderate': 0.6,
            'major': 0.9,
            'unknown': 0.7
        }
        
        return severity_scores.get(tamper_severity.lower(), 0.7)
    
    def compute_risk_score(self, behavior_score=None, classification_sensitivity=None,
                          classification_confidence=None, tamper_detected=None,
                          tamper_severity=None):
        """
        Compute fused risk score from multiple signals
        
        Args:
            behavior_score: Anomaly detection score
            classification_sensitivity: Document sensitivity level
            classification_confidence: Classification confidence
            tamper_detected: Tampering flag
            tamper_severity: Tampering severity
            
        Returns:
            Dictionary with risk score and components
        """
        components = {}
        
        # Behavioral risk
        if behavior_score is not None:
            components['behavior'] = self.normalize_anomaly_score(behavior_score)
        else:
            components['behavior'] = 0.0
        
        # Classification risk
        if classification_sensitivity is not None and classification_confidence is not None:
            components['classification'] = self.get_classification_risk(
                classification_sensitivity, classification_confidence
            )
        else:
            components['classification'] = 0.0
        
        # Integrity risk
        if tamper_detected is not None and tamper_severity is not None:
            components['integrity'] = self.get_integrity_risk(
                tamper_detected, tamper_severity
            )
        else:
            components['integrity'] = 0.0
        
        # Compute weighted sum
        risk_score = sum(
            components[k] * self.weights[k]
            for k in components.keys()
        )
        
        return {
            'risk_score': risk_score,
            'components': components
        }
    
    def create_alerts(self, risk_scores_df, threshold=0.6):
        """
        Create alerts for high-risk items
        
        Args:
            risk_scores_df: DataFrame with risk scores
            threshold: Risk score threshold for alerts
            
        Returns:
            DataFrame with alerts
        """
        alerts = risk_scores_df[risk_scores_df['risk_score'] >= threshold].copy()
        
        # Add alert metadata
        alerts['alert_id'] = [f"ALERT_{i+1:04d}" for i in range(len(alerts))]
        alerts['alert_timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        alerts['alert_priority'] = alerts['risk_score'].apply(
            lambda x: 'critical' if x >= 0.8 else 'high' if x >= 0.7 else 'medium'
        )
        
        return alerts
    
    def generate_alert_summary(self, alert):
        """
        Generate human-readable alert summary
        
        Args:
            alert: Alert row (Series)
            
        Returns:
            String summary
        """
        summary_parts = []
        
        if 'user_id' in alert:
            summary_parts.append(f"User {alert['user_id']}")
        
        if 'filename' in alert and pd.notna(alert['filename']):
            summary_parts.append(f"Document: {alert['filename']}")
        
        summary_parts.append(f"Risk Score: {alert['risk_score']:.2f}")
        
        # Add specific concerns
        concerns = []
        if 'behavior_risk' in alert and alert['behavior_risk'] > 0.5:
            concerns.append(f"Anomalous behavior (score: {alert['behavior_risk']:.2f})")
        
        if 'classification_risk' in alert and alert['classification_risk'] > 0.5:
            concerns.append(f"Sensitive document accessed")
        
        if 'integrity_risk' in alert and alert['integrity_risk'] > 0:
            concerns.append(f"Document tampering detected")
        
        if concerns:
            summary_parts.append("Concerns: " + "; ".join(concerns))
        
        return " | ".join(summary_parts)


def fuse_results(anomaly_path='outputs/anomaly_results.csv',
                classification_path='outputs/classification_results.csv',
                integrity_path='outputs/integrity_results.csv',
                output_path='outputs/risk_summary.csv'):
    """
    Main function to fuse all detection results into risk scores
    
    Args:
        anomaly_path: Path to anomaly detection results
        classification_path: Path to classification results
        integrity_path: Path to integrity results
        output_path: Path to save risk summary
    """
    print("="*60)
    print("RISK SCORING AND FUSION")
    print("="*60)
    
    # Load results
    print("\n1. Loading detection results...")
    anomaly_df = pd.read_csv(anomaly_path)
    classification_df = pd.read_csv(classification_path)
    integrity_df = pd.read_csv(integrity_path)
    
    print(f"  Anomaly results: {len(anomaly_df)} users")
    print(f"  Classification results: {len(classification_df)} documents")
    print(f"  Integrity results: {len(integrity_df)} documents")
    
    # Initialize scorer
    scorer = RiskScorer()
    
    # Compute user-level risks (behavioral + document access)
    print("\n2. Computing user-level risk scores...")
    user_risks = []
    
    for _, user_row in anomaly_df.iterrows():
        user_id = user_row['user_id']
        
        risk_result = scorer.compute_risk_score(
            behavior_score=user_row['anomaly_score']
        )
        
        user_risks.append({
            'entity_type': 'user',
            'entity_id': user_id,
            'risk_score': risk_result['risk_score'],
            'behavior_risk': risk_result['components']['behavior'],
            'classification_risk': 0.0,
            'integrity_risk': 0.0,
            'risk_level': user_row['risk_level'],
            'is_anomaly': user_row['is_anomaly']
        })
    
    # Compute document-level risks
    print("\n3. Computing document-level risk scores...")
    document_risks = []
    
    # Merge classification and integrity
    doc_df = classification_df.merge(
        integrity_df[['filename', 'tamper_detected', 'tamper_severity']],
        on='filename',
        how='left'
    )
    
    # Fill NaN values for documents without integrity checks
    doc_df.fillna({'tamper_detected': False, 'tamper_severity': 'none'}, inplace=True)
    
    for _, doc_row in doc_df.iterrows():
        risk_result = scorer.compute_risk_score(
            classification_sensitivity=doc_row['predicted_sensitivity'],
            classification_confidence=doc_row['confidence'],
            tamper_detected=doc_row['tamper_detected'],
            tamper_severity=doc_row['tamper_severity']
        )
        
        document_risks.append({
            'entity_type': 'document',
            'entity_id': doc_row['filename'],
            'filename': doc_row['filename'],
            'risk_score': risk_result['risk_score'],
            'behavior_risk': 0.0,
            'classification_risk': risk_result['components']['classification'],
            'integrity_risk': risk_result['components']['integrity'],
            'sensitivity': doc_row['predicted_sensitivity'],
            'tamper_detected': doc_row['tamper_detected']
        })
    
    # Combine all risks
    all_risks = pd.DataFrame(user_risks + document_risks)
    
    # Generate alerts
    print("\n4. Generating alerts...")
    alerts = scorer.create_alerts(all_risks, threshold=0.6)
    
    print(f"\nGenerated {len(alerts)} alerts:")
    print(f"  Critical: {sum(alerts['alert_priority'] == 'critical')}")
    print(f"  High: {sum(alerts['alert_priority'] == 'high')}")
    print(f"  Medium: {sum(alerts['alert_priority'] == 'medium')}")
    
    # Add alert summaries
    alerts['summary'] = alerts.apply(scorer.generate_alert_summary, axis=1)
    
    # Display top alerts
    print("\nTop 5 Alerts:")
    top_alerts = alerts.nlargest(5, 'risk_score')[
        ['alert_id', 'entity_type', 'entity_id', 'risk_score', 'alert_priority']
    ]
    print(top_alerts.to_string(index=False))
    
    # Save results
    print(f"\n5. Saving results...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    all_risks.to_csv(output_path, index=False)
    print(f"  Risk summary saved to {output_path}")
    
    alerts_path = output_path.replace('risk_summary.csv', 'alerts.csv')
    alerts.to_csv(alerts_path, index=False)
    print(f"  Alerts saved to {alerts_path}")
    
    # Save statistics
    stats = {
        'total_entities': len(all_risks),
        'users_evaluated': len(user_risks),
        'documents_evaluated': len(document_risks),
        'total_alerts': len(alerts),
        'alerts_by_priority': alerts['alert_priority'].value_counts().to_dict(),
        'average_risk_score': float(all_risks['risk_score'].mean()),
        'high_risk_entities': int((all_risks['risk_score'] >= 0.6).sum()),
        'fusion_weights': scorer.weights,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    stats_path = output_path.replace('risk_summary.csv', 'risk_statistics.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"  Statistics saved to {stats_path}")
    
    print("\n" + "="*60)
    print("RISK SCORING COMPLETE")
    print("="*60)
    
    return all_risks, alerts


if __name__ == '__main__':
    risk_summary, alerts = fuse_results()