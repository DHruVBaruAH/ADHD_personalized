# 🧠 ADHD Clinical Decision Support System (CDSS)

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![Status](https://img.shields.io/badge/Status-Prototype-green?style=for-the-badge)

**An AI-powered diagnostic suite combining Stacking Ensemble Learning, SHAP Explainability, and Computer Vision for objective ADHD assessment.**

[View Demo](#) • [Report Bug](#) • [Request Feature](#)

</div>

---

## 📋 Table of Contents
- [About The Project](#-about-the-project)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Tech Stack](#-tech-stack)
- [Installation & Setup](#-installation--setup)
- [Usage Guide](#-usage-guide)
- [Explainability (SHAP)](#-explainability-shap)
- [Future Scope](#-future-scope)
- [Disclaimer](#-disclaimer)

---

## 📖 About The Project

ADHD diagnosis has traditionally relied on subjective observation. This project bridges the gap between subjective behavior and objective data by analyzing **Continuous Performance Test (CPT-II)** metrics.

We engineered a **Stacking Ensemble Regressor** (combining LightGBM, XGBoost, and CatBoost) that processes over **360 clinical metrics** to predict an "ADHD Confidence Index" with **~94% accuracy**.

Beyond prediction, the system features a **Real-time Hyperactivity Monitor** that uses Computer Vision to track skeletal movement, providing a holistic view of the patient's behavior during testing.

---

## ✨ Key Features

| Feature | Description |
| :--- | :--- |
| **🤖 Multi-Model Stacking** | A "Council of Experts" (LGBM, XGB, CatBoost) weighted by a Meta-Learner (RidgeCV) for superior accuracy. |
| **📉 Leakage-Proof Pipeline** | Robust data engineering with Median Imputation, Yeo-Johnson Transformation, and Stratified Splitting. |
| **👁️ Computer Vision Guard** | Real-time skeletal tracking (MediaPipe) to detect fidgeting and hyperactivity via webcam. |
| **📊 Explainable AI (XAI)** | Integrated SHAP analysis (Beeswarm & Waterfall plots) to explain *why* a diagnosis was made. |
| **🏥 Clinical Dashboard** | Interactive Streamlit UI with "Ghost Patient" simulation and Radar Charts for symptom visualization. |

---

## 🏗 System Architecture

The model uses a Level-1 Stacking architecture designed to capture non-linear patterns in reaction time data.

![Architecture Diagram](refined_adhd_model_architecture_v2.png) 
*(Note: Replace with your generated architecture image path)*

---

## 🛠 Tech Stack

### **Core AI & ML**
* **Ensemble:** LightGBM, XGBoost, CatBoost
* **Meta-Learner:** RidgeCV
* **Preprocessing:** Scikit-Learn (Pipeline, ColumnTransformer, PowerTransformer)
* **Explainability:** SHAP (SHapley Additive exPlanations)

### **Application & Interface**
* **Dashboard:** Streamlit
* **Computer Vision:** OpenCV, MediaPipe
* **Data Handling:** Pandas, NumPy, Joblib

---

## ⚡ Installation & Setup

**Prerequisites:** Python 3.9 or higher.

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/ADHD-Prediction-System.git](https://github.com/YOUR_USERNAME/ADHD-Prediction-System.git)
    cd ADHD-Prediction-System
    ```

2.  **Create a Virtual Environment**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

---

## 🚀 Usage Guide

### 1. Running the Dashboard
Launch the main clinical interface:
```bash
python -m streamlit run app.py