from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# Initialize and train the model globally on server start
try:
    df = pd.read_csv('student_data.csv')
    df['study_hours'] = df['study_hours'].fillna(df['study_hours'].mean())
    df['attendance_percentage'] = df['attendance_percentage'].fillna(df['attendance_percentage'].mean())
    df['previous_grades'] = df['previous_grades'].fillna(df['previous_grades'].mean())
    df['tutoring'] = df['tutoring'].fillna(df['tutoring'].mode()[0])
    df['tutoring'] = df['tutoring'].map({'Yes': 1, 'No': 0})
    
    X = df[['study_hours', 'attendance_percentage', 'previous_grades', 'tutoring']]
    y = df['passed']
    
    model = LogisticRegression()
    model.fit(X, y)
    print("Model loaded and trained successfully.")
except Exception as e:
    print(f"Error initializing model: {e}")
    model = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not initialized'}), 500
        
    try:
        data = request.json
        hours = float(data.get('study_hours', 0))
        attendance = float(data.get('attendance_percentage', 0))
        grades = float(data.get('previous_grades', 0))
        
        tutor_val = str(data.get('tutoring', '0'))
        tutoring = 1 if tutor_val.lower() in ('1', 'yes', 'true') else 0
        
        user_data = pd.DataFrame({
            'study_hours': [hours],
            'attendance_percentage': [attendance],
            'previous_grades': [grades],
            'tutoring': [tutoring]
        })
        
        prediction = model.predict(user_data)
        result = int(prediction[0])
        
        return jsonify({'prediction': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
