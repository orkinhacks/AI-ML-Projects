import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split, KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error



df = pd.read_csv('Lifestyle Data.csv')
df = pd.get_dummies(df, columns=['Gender', 'Stress_Level'], drop_first=True)
X = df.drop('Healthy_Lifestyle_Score', axis=1).values
y = df['Healthy_Lifestyle_Score'].values
scaler = StandardScaler()
X = scaler.fit_transform(X)



def RandomForest(X, y):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    r2_scores = cross_val_score(model, X, y, cv=kf, scoring='r2')
    print(f'Cross-validated R2 Score: {np.mean(r2_scores) * 100:.2f}%')

    model.fit(X, y)
    return model



rfmodel = RandomForest(X, y)



def get_user_input():
    try:
        age = float(input("Enter your age: "))
        gender = input("Enter your gender (Male/Female): ").strip().capitalize()
        daily_steps = float(input("Enter your daily steps: "))
        calories_consumed = float(input("Enter calories consumed: "))
        sleep_hours = float(input("Enter your average sleep hours: "))
        water_intake_liters = float(input("Enter water intake (liters): "))
        stress_level = input("Enter your stress level (Low/Medium/High): ").strip().capitalize()
        exercise_hours = float(input("Enter your exercise hours: "))
        bmi = float(input("Enter your BMI: "))

        if gender not in ['Male', 'Female']:
            raise ValueError("Invalid gender input. Please enter 'Male' or 'Female'.")
        if stress_level not in ['Low', 'Medium', 'High']:
            raise ValueError("Invalid stress level input. Please enter 'Low', 'Medium', or 'High'.")

        gender_male = 1 if gender == 'Male' else 0

        stress_level_medium = 1 if stress_level == 'Medium' else 0
        stress_level_high = 1 if stress_level == 'High' else 0

        user_df = pd.DataFrame({
            'Age': [age],
            'Daily_Steps': [daily_steps],
            'Calories_Consumed': [calories_consumed],
            'Sleep_Hours': [sleep_hours],
            'Water_Intake_Liters': [water_intake_liters],
            'Exercise_Hours': [exercise_hours],
            'BMI': [bmi],
            'Gender_Male': [gender_male],
            'Stress_Level_Medium': [stress_level_medium],
            'Stress_Level_High': [stress_level_high]
        })

        user_input = scaler.transform(user_df.values)

        return user_input

    except ValueError as e:
        print(f"Input error: {e}")
        return None



def predict(user_input, model):
    if user_input is not None:
        prediction = model.predict(user_input)
        print(f'Predicted Healthy Lifestyle Score: {prediction[0]:.2f}')
    else:
        print("No prediction due to invalid input.")



user_input = get_user_input()
predict(user_input, rfmodel)