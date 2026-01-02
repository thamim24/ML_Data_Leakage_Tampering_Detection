"""
Document Integrity Verification
Checks document tampering using hash comparison and semantic similarity
"""
import pandas as pd
import numpy as np
import os
import json
import hashlib

# Optional imports for semantic similarity
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Note: sentence-transformers not installed. Using hash-only verification.")

class IntegrityVerifier:
    """Verify document integrity using hash and semantic similarity"""
    
    def __init__(self, use_semantic=False):
        """
        Initialize verifier
        
        Args:
            use_semantic: Use semantic similarity (requires sentence-transformers)
        """
        self.use_semantic = use_semantic and SENTENCE_TRANSFORMERS_AVAILABLE
        
        if self.use_semantic:
            print("Loading sentence transformer model...")
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            print("Model loaded successfully")
        else:
            self.model = None
            if use_semantic:
                print("Warning: sentence-transformers not available, using hash-only verification")
    
    def compute_hash(self, content):
        """
        Compute SHA-256 hash of content
        
        Args:
            content: Text or bytes content
            
        Returns:
            hex hash string
        """
        if isinstance(content, str):
            content = content.encode('utf-8')
        return hashlib.sha256(content).hexdigest()
    
    def compute_semantic_similarity(self, text1, text2):
        """
        Compute semantic similarity between two texts
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            similarity score (0-1)
        """
        if not self.use_semantic:
            return None
        
        # Compute embeddings
        embeddings = self.model.encode([text1, text2])
        
        # Compute cosine similarity
        similarity = np.dot(embeddings[0], embeddings[1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
        )
        
        return float(similarity)
    
    def verify_document(self, original_path, test_path=None, test_content=None):
        """
        Verify a document against its original
        
        Args:
            original_path: Path to original document
            test_path: Path to test document (optional)
            test_content: Test content as string (optional)
            
        Returns:
            Dictionary with verification results
        """
        # Read original
        with open(original_path, 'r', encoding='utf-8') as f:
            original_content = f.read()
        
        original_hash = self.compute_hash(original_content)
        
        # Read test document
        if test_path:
            with open(test_path, 'r', encoding='utf-8') as f:
                test_content = f.read()
        elif test_content is None:
            raise ValueError("Either test_path or test_content must be provided")
        
        test_hash = self.compute_hash(test_content)
        
        # Hash comparison
        hash_match = (original_hash == test_hash)
        
        # Semantic similarity (if enabled)
        semantic_sim = None
        if self.use_semantic and not hash_match:
            semantic_sim = self.compute_semantic_similarity(original_content, test_content)
        
        result = {
            'original_path': original_path,
            'test_path': test_path if test_path else 'content_provided',
            'hash_match': hash_match,
            'original_hash': original_hash,
            'test_hash': test_hash,
            'semantic_similarity': semantic_sim,
            'integrity_status': 'intact' if hash_match else 'modified',
            'tamper_detected': not hash_match
        }
        
        # Determine severity if tampered
        if not hash_match:
            if semantic_sim is not None:
                if semantic_sim > 0.95:
                    result['tamper_severity'] = 'minor'
                elif semantic_sim > 0.85:
                    result['tamper_severity'] = 'moderate'
                else:
                    result['tamper_severity'] = 'major'
            else:
                # Without semantic similarity, use simple classification
                result['tamper_severity'] = 'unknown'
        else:
            result['tamper_severity'] = 'none'
        
        return result
    
    def verify_all_documents(self, original_dir='data/documents',
                           tampered_dir='data/tampered_docs'):
        """
        Verify all documents in tampered directory against originals
        
        Args:
            original_dir: Directory with original documents
            tampered_dir: Directory with potentially tampered documents
            
        Returns:
            DataFrame with verification results
        """
        results = []
        
        print(f"\nVerifying documents from {tampered_dir}...")
        
        # Check each file in tampered directory
        if not os.path.exists(tampered_dir):
            print(f"Warning: {tampered_dir} does not exist. Creating verification for originals only.")
            tampered_files = []
        else:
            tampered_files = [f for f in os.listdir(tampered_dir) if f.endswith('.txt')]
        
        for filename in tampered_files:
            original_path = os.path.join(original_dir, filename)
            tampered_path = os.path.join(tampered_dir, filename)
            
            if not os.path.exists(original_path):
                print(f"Warning: No original found for {filename}")
                continue
            
            print(f"\nVerifying: {filename}")
            result = self.verify_document(original_path, test_path=tampered_path)
            result['filename'] = filename
            
            print(f"  Hash Match: {result['hash_match']}")
            if result['semantic_similarity'] is not None:
                print(f"  Semantic Similarity: {result['semantic_similarity']:.3f}")
            print(f"  Status: {result['integrity_status'].upper()}")
            print(f"  Tamper Severity: {result['tamper_severity'].upper()}")
            
            results.append(result)
        
        # Also verify original documents against themselves (baseline)
        print("\n\nBaseline verification of original documents...")
        original_files = [f for f in os.listdir(original_dir) if f.endswith('.txt')]
        
        for filename in original_files:
            if filename not in [r['filename'] for r in results]:
                original_path = os.path.join(original_dir, filename)
                
                print(f"\nBaseline: {filename}")
                result = self.verify_document(original_path, test_path=original_path)
                result['filename'] = filename
                result['test_path'] = 'self_verification'
                
                print(f"  Status: {result['integrity_status'].upper()}")
                
                results.append(result)
        
        return pd.DataFrame(results)


def verify_and_report(original_dir='data/documents',
                     tampered_dir='data/tampered_docs',
                     output_path='outputs/integrity_results.csv',
                     use_semantic=False):
    """
    Main function to verify document integrity
    
    Args:
        original_dir: Directory with original documents
        tampered_dir: Directory with potentially tampered documents
        output_path: Path to save results
        use_semantic: Use semantic similarity (requires sentence-transformers)
    """
    print("="*60)
    print("DOCUMENT INTEGRITY VERIFICATION")
    print("="*60)
    
    # Initialize verifier
    verifier = IntegrityVerifier(use_semantic=use_semantic)
    
    # Verify documents
    results = verifier.verify_all_documents(original_dir, tampered_dir)
    
    # Summary statistics
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)
    
    total_docs = len(results)
    tampered_docs = results['tamper_detected'].sum()
    intact_docs = total_docs - tampered_docs
    
    print(f"\nTotal documents verified: {total_docs}")
    print(f"Intact documents: {intact_docs}")
    print(f"Tampered documents: {tampered_docs}")
    
    if tampered_docs > 0:
        print("\nTamper severity distribution:")
        severity_counts = results[results['tamper_detected']]['tamper_severity'].value_counts()
        for severity, count in severity_counts.items():
            print(f"  {severity.capitalize()}: {count}")
        
        print("\nTampered documents:")
        tampered_df = results[results['tamper_detected']][
            ['filename', 'integrity_status', 'tamper_severity', 'semantic_similarity']
        ]
        print(tampered_df.to_string(index=False))
    
    # Save results
    print(f"\nSaving results to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results.to_csv(output_path, index=False)
    
    # Save summary
    summary = {
        'total_documents': int(total_docs),
        'intact_documents': int(intact_docs),
        'tampered_documents': int(tampered_docs),
        'tampering_rate': float(tampered_docs / total_docs) if total_docs > 0 else 0,
        'semantic_similarity_enabled': use_semantic
    }
    
    if tampered_docs > 0:
        summary['severity_distribution'] = results[results['tamper_detected']]['tamper_severity'].value_counts().to_dict()
    
    summary_path = output_path.replace('integrity_results.csv', 'integrity_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "="*60)
    print("INTEGRITY VERIFICATION COMPLETE")
    print("="*60)
    
    return results


if __name__ == '__main__':
    """
    Entry point for document integrity verification.
    Default: run with semantic similarity if possible,
    otherwise fall back to hash-only verification.
    """
    print("="*60)
    print("INITIALIZING DOCUMENT INTEGRITY VERIFICATION")
    print("="*60)

    if SENTENCE_TRANSFORMERS_AVAILABLE:
        # Run with semantic similarity enabled
        print("\nRunning with semantic similarity enabled...\n")
        results = verify_and_report(use_semantic=True)
        print("\nSemantic similarity analysis complete.")
    else:
        # Fallback: hash-only mode
        print("\nWarning: 'sentence-transformers' not installed.")
        print("Running in hash-only mode.\n")
        results = verify_and_report(use_semantic=False)

        # These lines now only appear in hash-only mode:
        print("\nNote: Running with hash-only verification for speed.")
        print("For semantic similarity analysis, install sentence-transformers:")
        print("  pip install sentence-transformers")
        print("Then run with use_semantic=True")

