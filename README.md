# 💰 Salary Predictor - Machine Learning Web Application

A user-friendly web app that predicts monthly salary based on experience, education level, and city tier using Machine Learning.

## ✨ Features
- Real-time salary prediction
- Clean and intuitive web interface built with Streamlit
- Trained using Linear Regression model
- Responsive design

## 🛠 Technologies Used
- **Python**
- **Scikit-learn** (Machine Learning)
- **Streamlit** (Frontend)
- **Pandas, NumPy**
- **Joblib** (Model deployment)

## 📊 Dataset & Model
- Generated realistic synthetic data (1000 samples)
- Features: Experience, Education Level, City Tier
- Model: Linear Regression
- R² Score: ~0.99

## 🚀 How to Run Locally

```bash
# Clone the repository
git clone https://github.com/bhavanishankar88/salary-predictor.git

# Go to project directory
cd salary-predictor

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run streamlit_app.py