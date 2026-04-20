# 🛸 EarthGuard - Asteroid Risk Prediction System

<div align="center">

![Version](https://img.shields.io/badge/version-2.0-orange)
![Python](https://img.shields.io/badge/Python-3.12-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28-red)
![License](https://img.shields.io/badge/License-MIT-green)

**An AI-powered early warning system for detecting potentially hazardous asteroids**

</div>

---

## 🌍 About EarthGuard

EarthGuard is an advanced **Machine Learning** based system that predicts whether an asteroid is potentially hazardous to Earth. Using orbital parameters and physical characteristics, our AI models achieve up to **94% accuracy** in risk assessment.

---

## 🚀 Features

| Feature | Description |
|---------|-------------|
| 🎯 **Real-time Risk Prediction** | Enter asteroid parameters and get instant risk assessment |
| 📊 **Interactive Visualizations** | Explore asteroid data with dynamic charts |
| 📁 **Batch Processing** | Upload CSV files for mass asteroid scanning |
| 🤖 **Multiple AI Models** | Random Forest, Decision Tree, Logistic Regression |
| 🎨 **Space-themed UI** | Immersive dark theme with glass morphism |

---

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **ML Models**: Scikit-learn (Random Forest, Decision Tree, Logistic Regression)
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Data Processing**: Pandas, NumPy

---

## 📁 Project Structure
EarthGuard/
├── app.py # Main Streamlit application
├── main.py # Pipeline entry point
├── notebooks/ # Jupyter notebooks for EDA & Modeling
│ ├── 01_eda.ipynb
│ ├── 02_preprocessing.ipynb
│ ├── 03_modeling.ipynb
│ └── 04_evaluation.ipynb
├── src/ # Source code modules
│ ├── preprocessing/ # Data cleaning & feature engineering
│ ├── training/ # Model training scripts
│ ├── evaluation/ # Metrics & comparison
│ └── utils/ # Logger & config utilities
├── models/ # Trained models (.pkl files)
├── data/ # Raw and processed data
├── reports/ # Results and evaluation metrics
└── requirements.txt # Dependencies

text

---

## 🚀 Quick Start

### Prerequisites
- Python 3.12 or higher
- pip package manager

### Installation

bash
# Clone the repository
git clone https://github.com/yourusername/EarthGuard.git
cd EarthGuard

# Install dependencies
pip install -r requirements.txt

# Run the web app
streamlit run app.py

## 🎯 How to Use
# 1. Single Asteroid Prediction
Select "RISK SCANNER" from sidebar

Enter orbital parameters (Eccentricity, Semi-major Axis, etc.)

Click "INITIATE RISK SCAN"

Get instant risk assessment with probability score

# 2. Batch Prediction
Select "MASS SCAN" from sidebar

Upload CSV file with asteroid data

Download results with risk scores

# 3. Data Visualization
Select "ASTEROID MAP" from sidebar

Explore interactive charts and distributions

# 📊 Model Performance
Model	             Accuracy	Precision	Recall	  F1-Score  
Random Forest	      94.2%	   93.8%	   92.5%	   93.1%   
Decision Tree	      91.5%	   90.2%	   89.8%	   90.0%  
Logistic Regression	89.8%	   88.5%	   87.2%	   87.8%  


# 👨‍🚀 Team
Role	Name
Lead Architect & AI Engineer -	Gouragopal Mohapatra  
Co-Developer & Data Scientist -	Arijit Kumar Mohanty  
Organization	StellarMind

# 📅 Version History
Version	Date	Changes  
v1.0	- April 2026	- Initial release with basic models  
v2.0	- April 2026	- Added Streamlit web interface  
v2.1	- April 2026	- Enhanced UI with space theme  
v2.2	- April 2026	- Added batch prediction  
v2.3	- April 2026	- Fixed 73-feature model compatibility

# 📞 Contact
For inquiries, collaborations, or support:

Email: contact@stellarmind.space

GitHub: github.com/stellarmind

# 🙏 Acknowledgments
NASA NEO Database for asteroid data

JPL Small-Body Database

ESA NEO Coordination Centre

# ⚠️ Disclaimer
This system is for research and monitoring purposes only. Always verify critical predictions with official space agencies.

<div align="center">
  ⚡ Made with ❤️ by <b>StellarMind</b> · 🌍 Protecting Earth since 2026
</div>
