import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib
import os

# Create models folder if not exists
os.makedirs('models', exist_ok=True)

# Generate realistic training data
np.random.seed(42)
n = 1000

experience = np.random.uniform(1, 20, n)
education_level = np.random.randint(1, 4, n)
city_tier = np.random.randint(1, 4, n)

# Realistic salary formula with noise
salary = 25000 + (experience * 2800) + (education_level * 9000) + (city_tier * 5000)
salary = salary + np.random.normal(0, 7000, n)   # adding noise

df = pd.DataFrame({
    'experience': experience,
    'education_level': education_level,
    'city_tier': city_tier,
    'salary': salary
})

# Train Model
X = df[['experience', 'education_level', 'city_tier']]
y = df['salary']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'models/salary_model.pkl')

print("✅ Model trained and saved successfully!")
print(f"Training Score: {model.score(X_train, y_train):.4f}")