"""
Document Classification using BERT
Classifies documents by sensitivity level: public, internal, confidential
"""
import pandas as pd
import numpy as np
import os
import json

# Optional imports for zero-shot classification
try:
    from transformers import pipeline
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Note: transformers not installed. Using keyword-based classification only.")

class DocumentClassifier:
    """Classify documents by sensitivity level"""
    
    def __init__(self, model_name='distilbert-base-uncased'):
        """
        Initialize classifier with pre-trained model
        
        Args:
            model_name: HuggingFace model name
        """
        self.model_name = model_name
        if TRANSFORMERS_AVAILABLE:
            self.device = 0 if torch.cuda.is_available() else -1
        else:
            self.device = -1
        
        # Sensitivity keywords for zero-shot classification
        self.sensitivity_keywords = {
            'public': ['announcement', 'public', 'general', 'all employees', 'everyone'],
            'internal': ['internal', 'employees only', 'staff', 'not for external', 'company use'],
            'confidential': ['confidential', 'restricted', 'secret', 'private', 'sensitive',
                           'classified', 'proprietary', 'financial', 'pii', 'personal data',
                           'gdpr', 'ccpa', 'unauthorized access', 'c-level', 'executive']
        }
        
        print(f"Initializing classifier with {model_name}")
        print(f"Using device: {'GPU' if self.device == 0 else 'CPU'}")
    
    def classify_by_keywords(self, text):
        """
        Simple keyword-based classification as baseline
        
        Args:
            text: Document text
            
        Returns:
            sensitivity level and confidence score
        """
        text_lower = text.lower()
        scores = {}
        
        for level, keywords in self.sensitivity_keywords.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            scores[level] = score
        
        if max(scores.values()) == 0:
            return 'internal', 0.5  # default to internal with low confidence
        
        predicted_level = max(scores, key=scores.get)
        confidence = scores[predicted_level] / (sum(scores.values()) + 1)
        
        return predicted_level, confidence
    
    def classify_zero_shot(self, text, max_length=512):
        """
        Zero-shot classification using pre-trained model
        
        Args:
            text: Document text
            max_length: Maximum text length for model
            
        Returns:
            sensitivity level and confidence score
        """
        if not TRANSFORMERS_AVAILABLE:
            print("Transformers not available, using keyword-based classification")
            return self.classify_by_keywords(text)
        
        # Truncate text if too long
        words = text.split()
        if len(words) > max_length:
            text = ' '.join(words[:max_length])
        
        try:
            # Use zero-shot classification
            classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=self.device
            )
            
            candidate_labels = ['public document', 'internal document', 'confidential document']
            result = classifier(text, candidate_labels)
            
            # Map labels to sensitivity levels
            label_map = {
                'public document': 'public',
                'internal document': 'internal',
                'confidential document': 'confidential'
            }
            
            predicted_label = result['labels'][0]
            confidence = result['scores'][0]
            sensitivity = label_map[predicted_label]
            
            return sensitivity, confidence
            
        except Exception as e:
            print(f"Zero-shot classification failed: {e}")
            print("Falling back to keyword-based classification")
            return self.classify_by_keywords(text)
    
    def classify_documents(self, doc_dir='data/documents', use_zero_shot=False):
        """
        Classify all documents in directory
        
        Args:
            doc_dir: Directory containing documents
            use_zero_shot: Use zero-shot model (slower but more accurate)
            
        Returns:
            DataFrame with classification results
        """
        results = []
        
        print(f"\nClassifying documents in {doc_dir}...")
        print(f"Method: {'Zero-shot BERT' if use_zero_shot else 'Keyword-based'}")
        
        for filename in os.listdir(doc_dir):
            if not filename.endswith('.txt'):
                continue
            
            filepath = os.path.join(doc_dir, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            print(f"\nProcessing: {filename}")
            
            if use_zero_shot:
                sensitivity, confidence = self.classify_zero_shot(content)
            else:
                sensitivity, confidence = self.classify_by_keywords(content)
            
            print(f"  -> Classified as: {sensitivity.upper()} (confidence: {confidence:.2f})")
            
            results.append({
                'filename': filename,
                'predicted_sensitivity': sensitivity,
                'confidence': confidence,
                'content_length': len(content),
                'word_count': len(content.split())
            })
        
        return pd.DataFrame(results)
    
    def evaluate_classification(self, results_df, metadata_path='data/document_metadata.csv'):
        """
        Evaluate classification against ground truth
        
        Args:
            results_df: Classification results
            metadata_path: Path to document metadata with true labels
            
        Returns:
            Evaluation metrics
        """
        # Load ground truth
        metadata = pd.read_csv(metadata_path)
        
        # Merge with predictions
        eval_df = results_df.merge(metadata[['filename', 'sensitivity']], on='filename')
        eval_df.rename(columns={'sensitivity': 'true_sensitivity'}, inplace=True)
        
        # Calculate accuracy
        eval_df['correct'] = (
            eval_df['predicted_sensitivity'] == eval_df['true_sensitivity']
        ).astype(int)
        
        accuracy = eval_df['correct'].mean()
        
        print("\n" + "="*60)
        print("CLASSIFICATION EVALUATION")
        print("="*60)
        print(f"\nOverall Accuracy: {accuracy:.2%}")
        
        print("\nPer-document results:")
        print(eval_df[['filename', 'true_sensitivity', 'predicted_sensitivity', 
                       'confidence', 'correct']].to_string(index=False))
        
        # Confusion matrix
        print("\nConfusion Matrix:")
        sensitivity_levels = ['public', 'internal', 'confidential']
        confusion = pd.crosstab(
            eval_df['true_sensitivity'],
            eval_df['predicted_sensitivity'],
            rownames=['True'],
            colnames=['Predicted']
        )
        print(confusion)
        
        return {
            'accuracy': float(accuracy),
            'correct_predictions': int(eval_df['correct'].sum()),
            'total_documents': len(eval_df),
            'per_document': eval_df.to_dict('records')
        }


def classify_and_evaluate(doc_dir='data/documents',
                         output_path='outputs/classification_results.csv',
                         use_zero_shot=False):
    """
    Main function to classify documents and evaluate
    
    Args:
        doc_dir: Directory containing documents
        output_path: Path to save results
        use_zero_shot: Use zero-shot model (requires internet and is slower)
    """
    print("="*60)
    print("DOCUMENT CLASSIFICATION")
    print("="*60)
    
    # Initialize classifier
    classifier = DocumentClassifier()
    
    # Classify documents
    results = classifier.classify_documents(doc_dir, use_zero_shot=use_zero_shot)
    
    # Evaluate
    evaluation = classifier.evaluate_classification(results)
    
    # Save results
    print(f"\nSaving results to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results.to_csv(output_path, index=False)
    
    # Save evaluation metrics
    eval_path = output_path.replace('classification_results.csv', 'classification_metrics.json')
    with open(eval_path, 'w') as f:
        json.dump(evaluation, f, indent=2)
    
    print("\n" + "="*60)
    print("CLASSIFICATION COMPLETE")
    print("="*60)
    
    return results, evaluation


if __name__ == '__main__':
    # Run classification with keyword-based method (fast, no API required)
    # Set use_zero_shot=True to use BERT model (requires transformers and model download)
    results, evaluation = classify_and_evaluate(use_zero_shot=True)
    
    print("\nNote: This uses keyword-based classification for speed.")
    print("For better accuracy, install transformers and run with use_zero_shot=True")
    print("This will download BART model (~1.6GB) on first run.")