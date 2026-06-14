# 💧 Egypt Water Quality Analysis & Prediction System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python"/>
  <img src="https://img.shields.io/badge/Power%20BI-F2C811?style=for-the-badge&logo=powerbi&logoColor=black"/>
  <img src="https://img.shields.io/badge/SQL%20Server-CC2927?style=for-the-badge&logo=microsoftsqlserver&logoColor=white"/>
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/Random%20Forest-228B22?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/SSRS-0078D4?style=for-the-badge&logo=microsoft&logoColor=white"/>
</p>

<p align="center">
  <b>An end-to-end data analytics and machine learning system for monitoring, analyzing, and predicting drinking water quality across Egypt's governorates and water companies.</b>
</p>

---

## 📌 Project Overview

This project was developed as part of the **ITI (Information Technology Institute) graduation project**, built on real data from the **Egyptian Water Company**. It combines business intelligence, data engineering, and machine learning to deliver a comprehensive water quality monitoring solution.

The system covers:
- 📊 **20 Power BI Dashboards** analyzing water quality, consumption, waste, companies, and governorates
- 📋 **6 SSRS Reports** covering operational KPIs across companies and usage metrics
- 🤖 **AI Prediction Model** (Random Forest, 99.3% accuracy) predicting water safety from 14 chemical parameters
- 💬 **GPT-powered Chatbot** for interactive water quality consultation
- 🗄️ **SQL Server Database** as the centralized data source for all analytics

---

## 🗂️ Project Structure

```
Egypt_Water_Quality_System/
│
├── 📁 DataBase/
│   ├── schema.sql                  # Database schema & table definitions
│   ├── stored_procedures.sql       # Stored procedures used across the system
│   └── sample_data.sql             # Sample/anonymized data for testing
│
├── 📁 Egypt_Water_Quality_ML/
│   ├── app2.py                     # Streamlit web application (main entry point)
│   ├── model_training.ipynb        # Model training notebook (EDA + Random Forest)
│   ├── water_quality_model.pkl     # Trained XGBoost model
│   ├── feature_names.pkl           # Feature names for model input
│   └── requirements.txt            # Python dependencies
│
├── 📁 Water_Quality_Project_SSRS/
│   ├── Report_Companies.rdl        # Company-level report
│   ├── Report_Consumption.rdl      # Water consumption report
│   ├── Report_Usage.rdl            # Usage analysis report
│   ├── Report_Waste.rdl            # Water waste report
│   ├── Report_Governorates.rdl     # Governorate-level report
│   └── Report_Summary.rdl          # Executive summary report
│
├── 📁 Presentation/
│   └── Egypt_Water_Quality.pptx    # Project presentation slides
│
├── Egypt Water Analysis.pbix       # Power BI file (20 dashboards)
└── README.md
```

---

## 📊 Power BI Dashboards (20 Dashboards)

Built using **Power BI Desktop**, connected directly to **SQL Server** via DirectQuery/Import mode.

| # | Dashboard | Description |
|---|-----------|-------------|
| 1 | Geographical Map | Interactive map showing water source and production distribution across Egypt |
| 2 | Source & Production | Water production volume analysis by source type |
| 3 | Production by Time | Time-series trends of water production across periods |
| 4 | Production by Region | Regional breakdown of water production capacity and output |
| 5 | Consumption | Water consumption patterns and volumes |
| 6 | Production vs Consumption | Comparative analysis of production output against consumption demand |
| 7 | Production vs Consumption (Detail) | Detailed drill-down of production/consumption gaps per company and region |
| 8 | Tests & Compliance | Overview of water quality test results against compliance thresholds |
| 9 | Quality & Compliance | WHO compliance rates across all monitored quality parameters |
| 10 | Quality, Compliance & Region | Regional breakdown of quality compliance scores |
| 11 | Coliform | Bacterial coliform contamination monitoring and trends |
| 12 | Coliform (Extended) | Detailed coliform analysis by station, source, and season |
| 13 | Turbidity | Turbidity (NTU) levels across stations and governorates |
| 14 | Station | Station-level performance and measurement tracking |
| 15 | Loss by Source | Water loss analysis traced back to production sources |
| 16 | Network Loss & Companies | Distribution network loss rates broken down by water company |
| 17 | Water Balance | System-wide water balance: input vs output vs loss |
| 18 | Loss by Region | Geographic breakdown of water loss across governorates |
| 19 | YOY Water Loss | Year-over-year comparison of water loss trends |
| 20 | Net Loss | Net water loss summary with KPI indicators |

---

## 📋 SSRS Reports (6 Reports)

Developed using **SQL Server Reporting Services (SSRS)**, pulling from the centralized SQL Server database.

| Report | Description |
|--------|-------------|
| Egypt Water Companies | Full directory and performance metrics of all Egyptian water distribution companies |
| Production Records | Detailed production volume records per station, source, and time period |
| Water Quality Compliance by Governorate | WHO compliance rates for all quality parameters broken down by governorate |
| Water Sources | Inventory and classification of all water sources (surface, groundwater, etc.) |
| Water Quality Test Records | Complete log of lab test results across all monitored parameters and stations |
| Egypt Water Sector — Executive Summary | High-level KPI summary for senior management and decision-makers |

---

## 🤖 Machine Learning Model

### Model Details

| Attribute | Value |
|-----------|-------|
| Algorithm | Random Forest Classifier |
| Accuracy | **99.3%** |
| Input Features | 14 chemical/physical parameters |
| Output | Binary — Safe (1) / Unsafe (0) |
| Training Data | Egyptian Water Company lab measurements |

### Input Parameters (14 Features)

| Parameter | Unit | WHO Safe Limit |
|-----------|------|----------------|
| pH | — | 6.5 – 8.5 |
| Turbidity | NTU | ≤ 5 |
| TDS | mg/L | ≤ 500 |
| Chlorine | mg/L | 0.2 – 0.8 |
| Electrical Conductivity | µS/cm | — |
| Nitrate | mg/L | ≤ 50 |
| Nitrite | mg/L | ≤ 0.1 |
| Ammonia | mg/L | ≤ 0.5 |
| Iron (Fe) | mg/L | ≤ 0.3 |
| Manganese (Mn) | mg/L | ≤ 0.1 |
| Sulfate | mg/L | ≤ 250 |
| Hardness | mg/L | ≤ 300 |
| Chloride | mg/L | ≤ 250 |
| Coliform | CFU/100mL | 0 |

---

## 🌐 Streamlit Web Application

An interactive web interface that allows users to:

1. **Input** the 14 water quality parameters from lab measurements
2. **Get an instant AI prediction** — Safe ✅ or Unsafe ❌ — with confidence score
3. **Identify pollution indicators** that exceed WHO thresholds
4. **Chat with a GPT-powered assistant** for detailed analysis, health risk explanation, and treatment recommendations

### Run Locally

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/egypt-water-quality-system.git
cd egypt-water-quality-system/Egypt_Water_Quality_ML

# Install dependencies
pip install -r requirements.txt

# Add your OpenAI API key in app2.py
# OPENAI_API_KEY = "your-key-here"

# Run the app
streamlit run app2.py
```

### Requirements

```
streamlit
pandas
scikit-learn
joblib
openai
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Database | Microsoft SQL Server |
| BI & Dashboards | Power BI Desktop |
| Reporting | SQL Server Reporting Services (SSRS) |
| ML Model | Python, Random Forest, Scikit-learn |
| Web App | Streamlit |
| AI Chatbot | OpenAI GPT-4o-mini API |
| Data Processing | Pandas, NumPy |
| Version Control | Git & GitHub |

---

## 📈 Key Insights from the Analysis

- Water quality varies significantly across governorates, with **Upper Egypt** showing higher risk levels
- **Turbidity and Iron** are the most frequent non-compliant parameters
- Seasonal patterns show **summer months** have elevated bacterial contamination rates
- Water waste rates average **~30%** across distribution networks, with notable regional differences
- Random Forest model achieves **99.3% accuracy**, enabling reliable real-time safety prediction

---

## 👨‍💻 Team

Developed by the **ITI Data Analysis Track** graduation project team.

> *"Leveraging data and AI to safeguard Egypt's most vital resource."*

---

## 📄 License

This project is for **educational and research purposes**. Data used belongs to the Egyptian Water Company and is used with permission for academic study.

---

<p align="center">
  Made with ❤️ for Egypt's Water Future 💧
</p>
