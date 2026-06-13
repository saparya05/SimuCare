# 🏥 SimuCare – Patient Digital Twin for Joint Clinical & Financial Risk Prediction

SimuCare is an AI-powered Digital Twin framework that integrates healthcare analytics and financial risk modeling to predict both medical and economic outcomes for patients.

By combining Electronic Health Records (EHRs), healthcare costs, and insurance data, SimuCare creates a virtual representation of a patient capable of forecasting clinical risks and financial impacts in real-time.

---

## 🚀 Features

### Clinical Risk Prediction
- ICU Stay Duration Prediction
- Hospital Readmission Risk Prediction
- Complication Risk Assessment
- Patient Outcome Forecasting

### Financial Risk Prediction
- Treatment Cost Estimation
- High-Cost Patient Identification
- Insurance Coverage Analysis
- Out-of-Pocket Expense Prediction

### Digital Twin Simulation
- Create a virtual patient profile
- Simulate multiple treatment scenarios
- Compare treatment outcomes and costs
- Support data-driven healthcare decisions

### Explainable AI
- Feature Importance Analysis
- SHAP Value Explanations
- Transparent Risk Scoring
- Interpretable Predictions

---

## 🧠 Problem Statement

Healthcare providers and insurance companies often operate independently when evaluating patient risks and financial outcomes.

This fragmented approach leads to:
- Poor resource allocation
- Increased healthcare costs
- Delayed interventions
- Inefficient decision-making

SimuCare bridges this gap by providing a unified AI framework capable of predicting both clinical and financial risks simultaneously.

---

## 🏗️ System Architecture

```text
Patient Data
(EHR + Insurance + Cost Data)
            │
            ▼
   Data Processing Pipeline
            │
            ▼
   Feature Engineering
            │
 ┌──────────┴──────────┐
 ▼                     ▼
Medical Models    Financial Models
 ▼                     ▼
ICU Risk         Cost Prediction
Readmission      Financial Risk
Complications    Insurance Analysis
 └──────────┬──────────┘
            ▼
      Digital Twin Engine
            ▼
   Scenario Simulation Dashboard
            ▼
 Clinical + Financial Insights
```

---

## 📊 Datasets

### Healthcare Dataset
**Source:** MIMIC-IV (MIT PhysioNet)

- 140 ICU Stays
- 20 Key Features
- Demographics
- Diagnoses
- Vital Signs
- Lab Results

**Objective**
- Predict Extended ICU Stay
- Predict Readmission Risk

---

### Financial Dataset
**Source:** Mendeley Health Insurance Dataset

- ~1300 Records
- Healthcare Costs
- Insurance Information
- Financial Indicators

**Objective**
- Predict High-Cost Patients
- Estimate Out-of-Pocket Expenses

---

## 🤖 Machine Learning Models

### Clinical Prediction Models
- Logistic Regression
- Random Forest
- XGBoost
- LightGBM
- LSTM (Future Scope)
- Transformers (Future Scope)

### Financial Prediction Models
- Linear Regression
- Logistic Regression
- Decision Trees
- Random Forest
- Gradient Boosting

---

## 📈 Results

### Healthcare Models

| Task | Best Model | Accuracy | AUC | F1 Score |
|--------|------------|------------|------------|------------|
| Extended ICU Stay | Random Forest | 86% | 0.93 | 0.74 |
| Readmission Risk | Random Forest | 84% | 0.65 | - |

### Financial Models

| Metric | Value |
|----------|---------|
| R² Score | 0.87 |
| MAE | ~3137 |

### Key Findings

✅ Random Forest achieved the highest performance for healthcare predictions.

✅ Financial models demonstrated strong cost forecasting capability.

✅ Combining clinical and financial analytics improved overall prediction reliability.

---

## 💻 Tech Stack

### Frontend
- React.js
- Tailwind CSS

### Backend
- Node.js
- Express.js
- MongoDB

### Machine Learning
- Python
- Scikit-Learn
- XGBoost
- Pandas
- NumPy

### Authentication
- JWT (JSON Web Tokens)

### API Testing
- Postman

---

## 📂 Project Structure

```text
SimuCare/
│
├── frontend/
│   ├── src/
│   ├── public/
│
├── backend/
│   ├── data/
│   ├── models/
│
├── README.md
│
└── requirements.txt
```

---

## ⚡ Installation

### Clone Repository

```bash
git clone https://github.com/saparya05/SimuCare.git
cd SimuCare
```

### Install Dependencies

Frontend:

```bash
cd frontend && npm install
```

Backend:

```bash
cd ../backend && npm install
```

### Configure Environment Variables

Create a `.env` file:

```env
MONGO_URI=<your_mongodb_uri>
JWT_SECRET=<your_secret>
PORT=5000
```

### Run Application

Backend:

```bash
npm run dev
```

Frontend:

```bash
npm start
```
---

## 🔄 Workflow

1. Collect patient and financial data
2. Clean and preprocess datasets
3. Engineer relevant features
4. Train clinical prediction models
5. Train financial prediction models
6. Integrate models into Digital Twin framework
7. Simulate treatment scenarios
8. Visualize outcomes through dashboard

---

## 🎯 Objectives

- Build a Patient Digital Twin framework
- Predict medical outcomes accurately
- Forecast financial risks proactively
- Enable treatment scenario simulations
- Improve healthcare decision-making
- Reduce overall healthcare costs

---

## 🔮 Future Work

- Real-time hospital data integration
- SHAP-based explainability dashboard
- IoT-enabled patient monitoring
- Multi-hospital deployment
- Expanded population datasets
- Integration with Hospital Information Systems (HIS)
- Advanced LSTM and Transformer-based prediction models

---

## 👨‍💻 Authors

### Saparya Jagannath
📧 saparya05@gmail.com

### Gauri Sharma
📧 gaurifsr@gmail.com

### Cheshta Arora
📧 cheshtaarora786@gmail.com

### Yashasvi Saini
📧 yashasvisaini355@gmail.com

**Department of Computer Science and Engineering**
Bharati Vidyapeeth's College of Engineering (Affiliated to GGSIPU, Delhi)

---

## 📜 License

This project is developed for academic and research purposes.
