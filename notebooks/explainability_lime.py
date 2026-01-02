"""
LIME Explainability for Document Classification
Provides interpretable explanations for text classification predictions
"""
import pandas as pd
import numpy as np
import os
import json

try:
    from lime.lime_text import LimeTextExplainer
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    print("Warning: LIME not installed. Install with: pip install lime")

class DocumentExplainer:
    """Generate LIME explanations for document classification"""
    
    def __init__(self, doc_dir='data/documents'):
        """
        Initialize explainer
        
        Args:
            doc_dir: Directory containing documents
        """
        if not LIME_AVAILABLE:
            raise ImportError("LIME is required. Install with: pip install lime")
        
        self.doc_dir = doc_dir
        self.class_names = ['public', 'internal', 'confidential']
        
        # Initialize LIME explainer
        self.explainer = LimeTextExplainer(class_names=self.class_names)
        
        print("LIME explainer initialized")
        print(f"Class names: {self.class_names}")
    
    def keyword_predictor(self, texts):
        """
        Simple keyword-based predictor for LIME
        Returns probability distribution over classes
        
        Args:
            texts: List of text samples
            
        Returns:
            Array of probability distributions
        """
        sensitivity_keywords = {
            'public': ['announcement', 'public', 'general', 'all employees', 'everyone'],
            'internal': ['internal', 'employees only', 'staff', 'not for external', 'company use'],
            'confidential': ['confidential', 'restricted', 'secret', 'private', 'sensitive',
                           'classified', 'proprietary', 'financial', 'pii', 'personal data',
                           'gdpr', 'ccpa', 'unauthorized access', 'c-level', 'executive']
        }
        
        results = []
        
        for text in texts:
            text_lower = text.lower()
            scores = {}
            
            # Count keyword matches for each class
            for class_name, keywords in sensitivity_keywords.items():
                score = sum(1 for kw in keywords if kw in text_lower)
                scores[class_name] = score
            
            # Convert to probabilities
            total = sum(scores.values())
            if total == 0:
                # Default to internal if no keywords found
                probs = [0.2, 0.6, 0.2]  # [public, internal, confidential]
            else:
                probs = [
                    scores['public'] / total,
                    scores['internal'] / total,
                    scores['confidential'] / total
                ]
            
            results.append(probs)
        
        return np.array(results)
    
    def explain_document(self, text, filename='document', num_features=10):
        """
        Generate LIME explanation for a document
        
        Args:
            text: Document text
            filename: Document filename for reference
            num_features: Number of features to show
            
        Returns:
            LIME explanation object
        """
        print(f"\nExplaining: {filename}")
        
        # Generate explanation
        exp = self.explainer.explain_instance(
            text,
            self.keyword_predictor,
            num_features=num_features,
            num_samples=500
        )
        
        # Get prediction
        probs = self.keyword_predictor([text])[0]
        predicted_class = self.class_names[np.argmax(probs)]
        confidence = np.max(probs)
        
        print(f"  Predicted: {predicted_class.upper()} (confidence: {confidence:.2f})")
        
        # Show top features
        print(f"  Top {min(5, num_features)} influential words:")
        for word, weight in exp.as_list()[:5]:
            direction = "confidential" if weight > 0 else "public"
            print(f"    '{word}': {weight:.3f} (pushes toward {direction})")
        
        return exp
    
    def save_explanation_html(self, exp, filename, output_dir='xai_outputs/lime_text_explanations'):
        """
        Save LIME explanation as HTML
        
        Args:
            exp: LIME explanation object
            filename: Original document filename
            output_dir: Output directory
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate HTML
        html = exp.as_html()
        
        # Save to file
        base_name = os.path.splitext(filename)[0]
        output_path = os.path.join(output_dir, f'{base_name}_explanation.html')
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        print(f"  HTML explanation saved to {output_path}")
        
        return output_path
    
    def save_explanation_json(self, exp, filename, predicted_class, confidence,
                             output_dir='xai_outputs/lime_text_explanations'):
        """
        Save LIME explanation as JSON
        
        Args:
            exp: LIME explanation object
            filename: Original document filename
            predicted_class: Predicted class
            confidence: Prediction confidence
            output_dir: Output directory
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract explanation data
        explanation_data = {
            'filename': filename,
            'predicted_class': predicted_class,
            'confidence': float(confidence),
            'top_features': [
                {'word': word, 'weight': float(weight)}
                for word, weight in exp.as_list()
            ],
            'class_probabilities': {
                class_name: float(prob)
                for class_name, prob in zip(self.class_names, exp.predict_proba)
            }
        }
        
        # Save to file
        base_name = os.path.splitext(filename)[0]
        output_path = os.path.join(output_dir, f'{base_name}_explanation.json')
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(explanation_data, f, indent=2)
        
        print(f"  JSON explanation saved to {output_path}")
        
        return output_path
    
    def explain_all_documents(self, doc_dir=None):
        """
        Generate explanations for all documents
        
        Args:
            doc_dir: Document directory (uses self.doc_dir if None)
            
        Returns:
            List of explanation results
        """
        if doc_dir is None:
            doc_dir = self.doc_dir
        
        results = []
        
        print(f"\nGenerating LIME explanations for documents in {doc_dir}...")
        
        for filename in os.listdir(doc_dir):
            if not filename.endswith('.txt'):
                continue
            
            filepath = os.path.join(doc_dir, filename)
            
            # Read document
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Generate explanation
            exp = self.explain_document(text, filename)
            
            # Get prediction
            probs = self.keyword_predictor([text])[0]
            predicted_class = self.class_names[np.argmax(probs)]
            confidence = np.max(probs)
            
            # Save explanations
            html_path = self.save_explanation_html(exp, filename)
            json_path = self.save_explanation_json(exp, filename, predicted_class, confidence)
            
            results.append({
                'filename': filename,
                'predicted_class': predicted_class,
                'confidence': confidence,
                'html_path': html_path,
                'json_path': json_path
            })
        
        return results


def generate_lime_explanations(doc_dir='data/documents',
                               output_dir='xai_outputs/lime_text_explanations'):
    """
    Main function to generate LIME explanations for all documents
    
    Args:
        doc_dir: Directory containing documents
        output_dir: Output directory for explanations
    """
    if not LIME_AVAILABLE:
        print("ERROR: LIME is not installed.")
        print("Install with: pip install lime")
        print("\nSkipping LIME explanations...")
        return None
    
    print("="*60)
    print("LIME EXPLAINABILITY FOR DOCUMENT CLASSIFICATION")
    print("="*60)
    
    try:
        # Initialize explainer
        explainer = DocumentExplainer(doc_dir)
        
        # Generate explanations for all documents
        results = explainer.explain_all_documents()
        
        # Save summary
        summary = {
            'total_documents': len(results),
            'output_directory': output_dir,
            'documents': results
        }
        
        summary_path = os.path.join(output_dir, 'lime_summary.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        
        print("\n" + "="*60)
        print("LIME EXPLANATIONS COMPLETE")
        print("="*60)
        print(f"\nGenerated explanations for {len(results)} documents")
        print(f"Outputs saved to {output_dir}/")
        print(f"Summary saved to {summary_path}")
        
        return results
        
    except Exception as e:
        print(f"\nError generating LIME explanations: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == '__main__':
    if LIME_AVAILABLE:
        results = generate_lime_explanations()
    else:
        print("\nTo use LIME explanations, install the package:")
        print("  pip install lime")
        print("\nNote: LIME is optional and not required for the prototype to run.")