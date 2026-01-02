# ML-Based Data Leakage and Tampering Detection Prototype

A **lightweight, explainable, multi-modal machine learning prototype** for detecting insider-driven data leakage and document tampering in enterprise workflows.

---

## 🎯 Overview

Modern enterprises face increasing risks from **authorized insider misuse**, where traditional rule-based Data Loss Prevention (DLP) systems fail to capture intent, context, and semantic manipulation. This prototype demonstrates how **machine learning and explainable AI (XAI)** can be combined to detect such threats in a transparent and practical manner.

The system integrates:

* **Behavioral anomaly detection** on user activity
* **NLP-based document sensitivity classification**
* **Document integrity verification** beyond simple hash checks
* **Risk fusion and alerting**
* **Explainable AI (XAI)** for analyst trust and auditability

This work is intended as a **research and demonstration prototype**, validated on synthetic enterprise data.

---

## 🔑 Core Contributions

This project makes the following contributions:

1. **Integrated Multi-Modal Detection**
   A unified pipeline that jointly analyzes **user behavior**, **document sensitivity**, and **semantic integrity**, rather than treating them as isolated security problems.

2. **Practical Risk Fusion Strategy**
   A configurable risk-scoring mechanism that combines heterogeneous ML outputs into a **single, interpretable risk score** suitable for enterprise alerting.

3. **Built-in Explainable AI (XAI)**
   Transparent explanations using **SHAP** (for behavioral anomalies) and **LIME** (for document classification), enabling analysts to understand *why* alerts are raised.

4. **Lightweight, Enterprise-Oriented Design**
   A modular, low-overhead architecture designed for feasibility in real enterprise environments, prioritizing interpretability and adaptability over black-box accuracy.

---

## 🧠 Detection Capabilities

### 1. Behavioral Anomaly Detection

* **Algorithm**: Isolation Forest
* **Objective**: Identify abnormal user behavior such as unusual access times, abnormal data volumes, or deviation from historical patterns
* **Output**: Anomaly scores, flagged users, behavioral risk indicators

### 2. Document Sensitivity Classification

* **Methods**:

  * Keyword-based classification (baseline, fast)
  * Optional BERT zero-shot classification (semantic, higher cost)
* **Classes**: Public, Internal, Confidential
* **Purpose**: Identify misuse or exposure of sensitive enterprise documents

### 3. Document Integrity Verification

* **Techniques**:

  * SHA-256 hash comparison
  * Optional semantic similarity analysis
* **Capability**: Detects subtle tampering where document meaning changes without obvious textual differences

### 4. Risk Scoring & Alerting

* **Fusion Inputs**: Behavioral risk, content sensitivity, integrity violations
* **Output**: Unified risk score (0–1), priority levels, and alerts

**Note on Risk Weights**
Fusion weights are empirically chosen to reflect common enterprise security priorities, where abnormal user behavior is often an early indicator of insider risk. These weights are configurable and can be learned or adapted in future work.

---

## 🔍 Explainable AI (XAI)

### SHAP – Behavioral Models

* Global feature importance across users
* Local explanations for individual anomaly decisions
* Answers: *“Why was this user flagged?”*

### LIME – Text Classification

* Word-level contribution analysis
* Interactive explanations for document sensitivity decisions
* Answers: *“Which terms made this document confidential?”*

Explainability is treated as a **first-class requirement**, not an afterthought.

---

## 📁 Project Structure

```
ML_DataLeakage_Prototype/
│
├── data/
│   ├── user_logs.csv
│   ├── log_features.csv
│   ├── document_metadata.csv
│   ├── documents/
│   └── tampered_docs/
│
├── notebooks/
│   ├── prototype_main.ipynb
│   ├── preprocessing.py
│   ├── anomaly_detection.py
│   ├── document_classification.py
│   ├── integrity_verification.py
│   ├── risk_scoring.py
│   ├── explainability_shap.py
│   └── explainability_lime.py
│
├── dashboard/
│   └── app.py
│
├── models/
│   ├── isolation_forest.pkl
│   └── behavioral_preprocessor.pkl
│
├── outputs/
│   ├── anomaly_results.csv
│   ├── classification_results.csv
│   ├── integrity_results.csv
│   ├── risk_summary.csv
│   └── alerts.csv
│
├── requirements.txt
└── README.md
```

---

## 🚀 Execution Options

### Option 1: End-to-End Notebook

Run the complete pipeline:

```bash
jupyter notebook notebooks/prototype_main.ipynb
```

### Option 2: Modular Execution

```bash
python notebooks/preprocessing.py
python notebooks/anomaly_detection.py
python notebooks/document_classification.py
python notebooks/integrity_verification.py
python notebooks/risk_scoring.py
```

### Option 3: Interactive Dashboard

```bash
streamlit run dashboard/app.py
```

---

## 📊 Experimental Summary (Synthetic Data)

| Metric              | Observation                                  |
| ------------------- | -------------------------------------------- |
| Dataset             | Synthetic enterprise logs and documents      |
| Anomalous Users     | ~10% (controlled contamination)              |
| Sensitive Documents | Detected via NLP classification              |
| Tampered Documents  | Identified via hash + semantic checks        |
| Overall Outcome     | Effective detection with low false positives |

**Important**:
Results reflect **controlled synthetic conditions** and are presented to demonstrate feasibility, not real-world performance guarantees.

---

## 🔐 Security & Ethical Considerations

* No real PII or confidential enterprise data is used
* XAI outputs may expose sensitive patterns and must be access-controlled
* Not production-hardened; intended for research and demonstration only

---

## 🧪 Limitations & Future Work

* Evaluation limited to synthetic data
* Risk fusion weights are heuristic
* No real-time streaming integration
* No federated or privacy-preserving learning

**Future extensions** include:

* Adaptive or learned risk fusion
* Real-time monitoring
* Federated learning for privacy
* Integration with SIEM/SOC pipelines

---

## 📚 References

* Liu et al., *Isolation Forest*, 2008
* Lundberg & Lee, *SHAP*, 2017
* Ribeiro et al., *LIME*, 2016

---

## 📌 Status

**Project Type**: Research Prototype
**Focus**: Explainable ML for Enterprise Security
**Readiness**: Demonstration & academic evaluation

---