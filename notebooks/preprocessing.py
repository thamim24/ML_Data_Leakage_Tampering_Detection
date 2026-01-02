"""
Preprocessing utilities for ML Data Leakage Prototype
Handles feature engineering for behavioral data and document embeddings
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os

class BehavioralPreprocessor:
    """Preprocess user log data for anomaly detection"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def create_features(self, df):
        """Create behavioral features from raw logs"""
        df = df.copy()
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Time-based features
        df['is_night'] = (df['hour'] < 6) | (df['hour'] > 22)
        df['is_weekend'] = df['day_of_week'].isin([5, 6])
        
        # Aggregate features per user per day
        df['date'] = df['timestamp'].dt.date
        
        # Create user-day aggregates
        user_day_agg = df.groupby(['user_id', 'date']).agg({
            'data_volume_mb': ['sum', 'mean', 'max'],
            'num_files_accessed': ['sum', 'mean', 'max'],
            'login_attempts': 'sum',
            'failed_logins': 'sum',
            'session_duration_min': ['sum', 'mean', 'max'],
            'is_night': 'sum',
            'is_weekend': 'max'
        }).reset_index()
        
        # Flatten column names
        user_day_agg.columns = ['_'.join(col).strip('_') for col in user_day_agg.columns.values]
        
        # Calculate user statistics (rolling features)
        user_stats = df.groupby('user_id').agg({
            'data_volume_mb': ['mean', 'std', 'max'],
            'num_files_accessed': ['mean', 'std'],
            'session_duration_min': ['mean', 'std'],
            'failed_logins': 'sum'
        })
        
        user_stats.columns = ['user_' + '_'.join(col).strip('_') for col in user_stats.columns.values]
        user_stats = user_stats.reset_index()
        
        # Count unique locations and devices per user
        location_counts = df.groupby('user_id')['location'].nunique().reset_index()
        location_counts.columns = ['user_id', 'unique_locations']
        
        device_counts = df.groupby('user_id')['device'].nunique().reset_index()
        device_counts.columns = ['user_id', 'unique_devices']
        
        # Merge all features
        user_features = user_stats.merge(location_counts, on='user_id')
        user_features = user_features.merge(device_counts, on='user_id')
        
        return user_features
    
    def encode_categorical(self, df, categorical_cols):
        """Encode categorical variables"""
        df = df.copy()
        
        for col in categorical_cols:
            if col in df.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
                else:
                    df[col] = self.label_encoders[col].transform(df[col].astype(str))
        
        return df
    
    def scale_features(self, df, fit=True):
        """Scale numerical features"""
        if fit:
            scaled_data = self.scaler.fit_transform(df)
        else:
            scaled_data = self.scaler.transform(df)
        
        return pd.DataFrame(scaled_data, columns=df.columns, index=df.index)
    
    def save(self, path='models/behavioral_preprocessor.pkl'):
        """Save preprocessor"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
    
    @staticmethod
    def load(path='models/behavioral_preprocessor.pkl'):
        """Load preprocessor"""
        return joblib.load(path)


class DocumentPreprocessor:
    """Preprocess documents for classification and integrity checking"""
    
    def __init__(self):
        pass
    
    def load_documents(self, doc_dir='data/documents'):
        """Load all documents from directory"""
        documents = []
        
        for filename in os.listdir(doc_dir):
            if filename.endswith('.txt'):
                filepath = os.path.join(doc_dir, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                documents.append({
                    'filename': filename,
                    'content': content,
                    'filepath': filepath
                })
        
        return pd.DataFrame(documents)
    
    def clean_text(self, text):
        """Basic text cleaning"""
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text
    
    def chunk_text(self, text, max_length=512):
        """Chunk long text for BERT models"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), max_length):
            chunk = ' '.join(words[i:i+max_length])
            chunks.append(chunk)
        
        return chunks


def create_log_features(logs_path='data/user_logs.csv', output_path='data/log_features.csv'):
    """Main function to create and save log features"""
    print("Loading user logs...")
    df = pd.read_csv(logs_path)
    
    print("Creating behavioral features...")
    preprocessor = BehavioralPreprocessor()
    features = preprocessor.create_features(df)
    
    print(f"Created {len(features.columns)} features for {len(features)} users")
    
    # Save features
    features.to_csv(output_path, index=False)
    print(f"Saved features to {output_path}")
    
    # Save preprocessor
    preprocessor.save()
    print("Saved preprocessor to models/behavioral_preprocessor.pkl")
    
    return features


if __name__ == '__main__':
    # Create behavioral features
    features = create_log_features()
    print("\nFeature columns:")
    print(features.columns.tolist())
    print("\nFeature statistics:")
    print(features.describe())