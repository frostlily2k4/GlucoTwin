# 🩺 GlucoTwin AI
### A Predictive Digital Twin for Type 2 Diabetes

**Happiest Health — Digital Twin Challenge 2026**

> From data to healthier possibilities: exploring how machine learning and interactive simulation can help anticipate short-term glucose risk.

---

## 📌 1. Project Overview

GlucoTwin AI is a proof-of-concept digital health application that combines synthetic patient data, machine-learning-based glucose-risk estimation, and an interactive what-if simulator.

The application explores whether glucose may exceed **180 mg/dL within the next two hours** and allows users to compare model-generated estimates under different meal, post-meal walking, and sleep scenarios.

The project demonstrates a potential approach to personalized digital health exploration. It is an experimental prototype and has not been validated for clinical use.

## 🎯 2. Problem Statement

Healthcare monitoring often focuses on observing measurements after changes have occurred. Exploring potential future risk using available health information could help researchers investigate more proactive approaches to health monitoring.

**Challenge:** How can a digital twin prototype combine patient-related information and machine learning to explore short-term glucose risk and hypothetical lifestyle scenarios?

## 🏥 3. Healthcare Use Case

**Domain:** Type 2 diabetes and glucose-risk monitoring.

GlucoTwin AI provides an interactive environment to explore:

- Glucose trends and synthetic patient information.
- Machine-learning-based estimates of short-term glucose risk.
- A prediction target of glucose exceeding 180 mg/dL within two hours.
- Hypothetical meal, post-meal walking, and sleep scenarios.
- Patient health indicators in a unified dashboard.

**Intended users:** Researchers, students, and developers exploring digital health technologies.

**Important:** The application is not intended to diagnose diabetes, recommend treatment, or replace healthcare professionals.

## ✨ 4. Key Features

- **Patient Dashboard:** Displays synthetic patient information and health indicators.
- **Glucose Trend Visualization:** Presents glucose-related trends for exploration.
- **Risk Prediction:** Estimates the likelihood of a glucose-threshold event using a machine-learning model.
- **What-If Simulator:** Allows users to adjust meal, walking, and sleep settings and compare predicted outputs.
- **Interactive Visualizations:** Presents model outputs in an accessible dashboard.
- **Digital Twin Concept:** Combines patient-related information, prediction, and scenario exploration in one prototype.

## 🧠 5. AI/ML Model Details

**Model approach:** Gradient-boosted trees.

**Prediction objective:** Estimate whether glucose may exceed 180 mg/dL within the next two hours.

**Data:** Synthetic patient data used for development and evaluation.

**Reported evaluation metric:** AUROC 0.96, as reported by the project's synthetic-data evaluation.

The reported metric is preliminary and does not establish performance on real-world patients. Independent validation, representative clinical data, and appropriate evaluation would be required before considering clinical use.

### Model workflow

1. Prepare synthetic patient data.
2. Preprocess inputs and prepare model features.
3. Train and evaluate the machine-learning model.
4. Generate glucose-risk estimates.
5. Display predictions and hypothetical scenario comparisons in the dashboard.

## 🏗️ 6. System Architecture

The prototype follows this conceptual workflow:

```text
Synthetic Patient Data
          |
          v
Data Preprocessing
          |
          v
Feature Preparation
          |
          v
Gradient-Boosted Tree Model
          |
          v
Short-Term Glucose-Risk Estimate
          |
          v
Streamlit Dashboard
          |
          v
Glucose Trends + What-If Simulator
```

**Architecture diagram:** [Add the path or public URL of your architecture PDF here.]

## 🛠️ 7. Technical Stack

| Component | Technology |
|---|---|
| Programming language | Python |
| Dashboard framework | Streamlit |
| Machine learning | Gradient-boosted tree model |
| Data processing | Add the libraries used by your project |
| Visualization | Add the charting libraries used by your project |
| Development environment | Python virtual environment |

**Note:** Confirm the exact libraries and versions from your project's `requirements.txt` before submission.

## 🚀 8. Installation and Setup

### Prerequisites

- Python 3.11.9, as used in the development environment.
- Git.
- A terminal or command prompt.

### Step 1: Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/GlucoTwin-AI.git
cd GlucoTwin-AI
```

Replace `frostlily2k4` with your GitHub username.

### Step 2: Create a virtual environment

**Windows:**

```bash
py -3.11 -m venv .venv
.venv\Scripts\activate
```

### Step 3: Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Train the model if required

If your project requires model training before the application starts, run:

```bash
python src/train.py
```

Ensure that the script and output paths match the actual files in your repository.

### Step 5: Launch the dashboard

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal.

## 🎬 9. Demo Video

**Demo video:** [ADD YOUR UNLISTED YOUTUBE VIDEO LINK HERE]

The demonstration should cover:

1. Project introduction.
2. Patient dashboard and glucose trends.
3. Short-term glucose-risk estimate.
4. Meal, walking, and sleep scenario simulation.
5. Limitations and future scope.

The video must be uploaded to YouTube as **Unlisted**, and the link must be accessible to reviewers. Add the same link here after uploading.

## 📊 10. Results and Outcomes

The current proof of concept demonstrates:

- Integration of a machine-learning model with an interactive dashboard.
- Visualization of synthetic patient information and glucose-related trends.
- Short-term glucose-risk estimation.
- Interactive exploration of hypothetical lifestyle scenarios.

**Reported result:** AUROC 0.96 on the project's synthetic-data evaluation setup.

This result must not be interpreted as evidence of clinical effectiveness or real-world predictive accuracy. The prototype requires further independent testing and validation.

## 🔮 11. Future Scope

- Evaluate performance on ethically sourced, representative clinical datasets.
- Assess generalization and prediction calibration.
- Explore securely integrated wearable and electronic health record data.
- Improve transparency and explainability of predictions.
- Evaluate usability with appropriate healthcare professionals.
- Investigate privacy, security, and responsible deployment requirements.

These are proposed future improvements, not established capabilities of the current prototype.

## 👥 12. Team Details

- **Team name:** PartnerHealer
- **Team size:** Solo
- **Team member:** Jothi A
- **Role:** Project development, machine learning, and dashboard implementation, as applicable.
- **GitHub profile:** https://github.com/frostlily2k4

## 🎓 13. College / Incubator Information

- **College / Institution:** AVP COLLEGE OF ARTS AND SCIENCE
- **Department:** B.Sc AI &  ML

## 📄 14. Project Presentation

**Presentation:** https://drive.google.com/file/d/1wQnUZOVyn-lEZc8jY9o-my0QX2xJXhji/view?usp=sharing

The presentation covers the problem statement, proposed solution, architecture, machine-learning approach, dashboard, evaluation, limitations, and future scope.

## 🧩 15. Architecture Document

**Architecture PDF:** https://drive.google.com/file/d/1HKyN24oGDdtD0JPadE1rym_nIQCjz3xi/view?usp=sharing

Ensure that the document is uploaded to the repository or another publicly accessible location.

## ⚖️ 16. Open-Source License

**License:** [Specify the license selected for this repository.]

If you choose the MIT License, add a `LICENSE` file containing the appropriate MIT license text and your copyright information. Do not claim a license until you have added it.

## 🔐 17. Data Privacy and Responsible AI

- The project uses synthetic data for its proof of concept.
- Do not upload real patient records, credentials, API keys, or other sensitive information.
- Predictions and simulated outcomes are model estimates.
- Clinical deployment would require appropriate validation, privacy safeguards, security controls, and human oversight.

## ⚠️ 18. Disclaimer

GlucoTwin AI is an educational and experimental proof of concept. It is not a medical device and is not intended for diagnosis, treatment, or clinical decision-making. Synthetic-data evaluation does not establish clinical validity.

## 🙏 19. Acknowledgement

Developed for the **Happiest Health Digital Twin Challenge 2026**.

---

**Project:** GlucoTwin AI  
**Focus:** Predictive digital health and Type 2 diabetes  
**Status:** Proof of concept  
**Data:** Synthetic  
