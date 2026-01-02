"""
Generate synthetic data for ML Data Leakage Prototype
Creates user logs, documents, and tampered documents
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import hashlib

np.random.seed(42)

# Generate synthetic user logs
def generate_user_logs(n_users=50, n_logs_per_user=100):
    """Generate synthetic user activity logs with normal and anomalous patterns"""
    
    users = [f"U{str(i).zfill(3)}" for i in range(1, n_users + 1)]
    data = []
    
    for user in users:
        # Decide if user is anomalous (10% chance)
        is_anomalous = np.random.random() < 0.1
        
        for _ in range(n_logs_per_user):
            timestamp = datetime.now() - timedelta(days=np.random.randint(0, 30),
                                                   hours=np.random.randint(0, 24),
                                                   minutes=np.random.randint(0, 60))
            
            # Normal behavior patterns
            if not is_anomalous:
                hour = np.random.choice(range(9, 18), p=[0.05, 0.1, 0.15, 0.2, 0.2, 0.15, 0.1, 0.03, 0.02])
                data_volume_mb = np.random.gamma(2, 5)  # skewed towards smaller transfers
                num_files = np.random.poisson(5)
                login_attempts = 1
                failed_logins = 0
                location = np.random.choice(['Office_A', 'Office_B', 'Remote_VPN'], p=[0.6, 0.3, 0.1])
                device = np.random.choice(['Laptop_A', 'Desktop_A', 'Laptop_B'], p=[0.5, 0.3, 0.2])
            else:
                # Anomalous patterns
                hour = np.random.choice([0, 1, 2, 3, 22, 23] + list(range(9, 18)))
                data_volume_mb = np.random.gamma(5, 20)  # larger transfers
                num_files = np.random.poisson(15)
                login_attempts = np.random.choice([1, 2, 3, 4], p=[0.6, 0.2, 0.15, 0.05])
                failed_logins = np.random.choice([0, 1, 2], p=[0.7, 0.2, 0.1])
                location = np.random.choice(['Office_A', 'Office_B', 'Remote_VPN', 'Unknown'], p=[0.3, 0.2, 0.3, 0.2])
                device = np.random.choice(['Laptop_A', 'Desktop_A', 'Laptop_B', 'Unknown_Device'], p=[0.3, 0.2, 0.2, 0.3])
            
            timestamp = timestamp.replace(hour=hour)
            
            data.append({
                'user_id': user,
                'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                'hour': hour,
                'day_of_week': timestamp.weekday(),
                'data_volume_mb': round(data_volume_mb, 2),
                'num_files_accessed': num_files,
                'login_attempts': login_attempts,
                'failed_logins': failed_logins,
                'location': location,
                'device': device,
                'session_duration_min': round(np.random.exponential(30), 2),
                'is_anomalous': 1 if is_anomalous else 0
            })
    
    df = pd.DataFrame(data)
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df

# Generate synthetic documents
def generate_documents():
    """Generate synthetic documents with different sensitivity levels"""
    
    documents = {
        'public_announcement.txt': {
            'content': """Company Annual Meeting Announcement
            
We are pleased to announce our annual company meeting will be held on December 15th.
All employees are invited to attend. The meeting will cover company achievements,
future goals, and team building activities. Location: Main Conference Hall.
Time: 10:00 AM - 4:00 PM. Lunch will be provided.

Please RSVP by December 1st to hr@company.com.""",
            'sensitivity': 'public'
        },
        
        'finance_report.txt': {
            'content': """Q3 Financial Report - CONFIDENTIAL

Total Revenue: $15.2M (up 12% YoY)
Operating Expenses: $8.4M
Net Income: $6.8M
EBITDA: $7.2M

Key Growth Areas:
- Enterprise Sales: +25%
- Cloud Services: +35%
- International Markets: +18%

Financial Projections Q4:
Expected Revenue: $17.5M
Target Net Income: $8.1M

Sensitive Information: Cash reserves $42M
Credit Line: $25M available

This document contains confidential financial information.
Distribution restricted to C-level executives only.""",
            'sensitivity': 'confidential'
        },
        
        'hr_policy.txt': {
            'content': """Human Resources Policy - Internal Use Only

Employee Benefits Overview:
- Health Insurance: Full coverage for employees and dependents
- 401(k): Company matches up to 5%
- Paid Time Off: 20 days annually
- Sick Leave: 10 days annually

Performance Review Process:
Reviews conducted bi-annually. Managers assess performance against goals.
Salary adjustments based on performance ratings and market conditions.

Disciplinary Procedures:
Minor infractions: Verbal warning
Repeated issues: Written warning
Serious violations: Immediate termination

This document is for internal use and should not be shared externally.""",
            'sensitivity': 'internal'
        },
        
        'customer_database.txt': {
            'content': """Customer Database Extract - HIGHLY CONFIDENTIAL

Customer Records (Sample):
ID: C001, Name: Acme Corp, Email: contact@acme.com, Phone: 555-0100
Revenue: $250K annually, Contract: Enterprise Plan, Renewal: 2024-06-15

ID: C002, Name: TechStart Inc, Email: info@techstart.com, Phone: 555-0200
Revenue: $180K annually, Contract: Professional Plan, Renewal: 2024-08-20

ID: C003, Name: Global Industries, Email: sales@global.com, Phone: 555-0300
Revenue: $420K annually, Contract: Enterprise Plus, Renewal: 2024-12-01

CRITICAL: This data contains PII and is protected under GDPR/CCPA.
Unauthorized access or distribution is prohibited and may result in
legal action and termination of employment.""",
            'sensitivity': 'confidential'
        },
        
        'marketing_plan.txt': {
            'content': """Marketing Strategy 2024 - Internal Document

Target Markets:
1. Enterprise sector (Fortune 500)
2. Mid-market technology companies
3. Healthcare industry
4. Financial services

Campaign Strategies:
- Q1: Brand awareness through social media and trade shows
- Q2: Product launch with webinar series
- Q3: Customer success stories and case studies
- Q4: Holiday promotion and year-end discounts

Budget Allocation:
Digital Marketing: 40%
Events & Trade Shows: 25%
Content Marketing: 20%
PR & Media: 15%

Internal use only - not for external distribution.""",
            'sensitivity': 'internal'
        }
    }
    
    os.makedirs('data/documents', exist_ok=True)
    os.makedirs('data/tampered_docs', exist_ok=True)
    
    # Write original documents
    for filename, doc_info in documents.items():
        filepath = os.path.join('data/documents', filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(doc_info['content'])
    
    # Create tampered versions
    tampered_docs = {
        'finance_report.txt': """Q3 Financial Report - CONFIDENTIAL

Total Revenue: $15.2M (up 12% YoY)
Operating Expenses: $8.4M
Net Income: $6.8M
EBITDA: $7.2M

Key Growth Areas:
- Enterprise Sales: +25%
- Cloud Services: +35%
- International Markets: +18%

Financial Projections Q4:
Expected Revenue: $18.9M [MODIFIED]
Target Net Income: $9.5M [MODIFIED]

Sensitive Information: Cash reserves $42M
Credit Line: $25M available

This document contains confidential financial information.
Distribution restricted to C-level executives only.""",
        
        'customer_database.txt': """Customer Database Extract - HIGHLY CONFIDENTIAL

Customer Records (Sample):
ID: C001, Name: Acme Corp, Email: contact@acme.com, Phone: 555-0100
Revenue: $250K annually, Contract: Enterprise Plan, Renewal: 2024-06-15

ID: C002, Name: TechStart Inc, Email: info@techstart.com, Phone: 555-0200
Revenue: $180K annually, Contract: Professional Plan, Renewal: 2024-08-20

[DELETED SECTION]

CRITICAL: This data contains PII and is protected under GDPR/CCPA.
Unauthorized access or distribution is prohibited."""
    }
    
    for filename, content in tampered_docs.items():
        filepath = os.path.join('data/tampered_docs', filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
    
    # Create metadata CSV
    metadata = []
    for filename, doc_info in documents.items():
        filepath = os.path.join('data/documents', filename)
        with open(filepath, 'rb') as f:
            content_hash = hashlib.sha256(f.read()).hexdigest()
        
        metadata.append({
            'filename': filename,
            'sensitivity': doc_info['sensitivity'],
            'hash': content_hash,
            'created_date': datetime.now().strftime('%Y-%m-%d'),
            'size_bytes': os.path.getsize(filepath)
        })
    
    metadata_df = pd.DataFrame(metadata)
    metadata_df.to_csv('data/document_metadata.csv', index=False)
    
    print(f"Generated {len(documents)} documents")
    print(f"Generated {len(tampered_docs)} tampered documents")

if __name__ == '__main__':
    print("Generating synthetic data...")
    
    # Determine base directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)  # parent directory
    data_dir = os.path.join(base_dir, 'data')
    
    # Generate user logs
    print("\n1. Generating user logs...")
    logs_df = generate_user_logs(n_users=50, n_logs_per_user=100)
    logs_df.to_csv(os.path.join(data_dir, 'user_logs.csv'), index=False)
    print(f"Generated {len(logs_df)} log entries for {logs_df['user_id'].nunique()} users")
    print(f"Anomalous users: {logs_df[logs_df['is_anomalous']==1]['user_id'].nunique()}")
    
    # Generate documents
    print("\n2. Generating documents...")
    generate_documents()
    
    print("\n[SUCCESS] Data generation complete!")
    print("\nGenerated files:")
    print(f"  - {os.path.join(data_dir, 'user_logs.csv')}")
    print(f"  - {os.path.join(data_dir, 'document_metadata.csv')}")
    print(f"  - {os.path.join(data_dir, 'documents')} (5 documents)")
    print(f"  - {os.path.join(data_dir, 'tampered_docs')} (2 tampered documents)")