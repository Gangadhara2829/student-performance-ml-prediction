import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def main():
    print("--- Student Performance Predictor ---")
    
    # 1. Load dataset
    print("\nLoading dataset...")
    try:
        df = pd.read_csv('student_data.csv')
        print("Dataset loaded successfully.")
    except FileNotFoundError:
        print("Error: student_data.csv not found.")
        return

    # 2. Preprocess data
    print("Preprocessing data...")
    
    # Handle missing values by filling numeric with mean and categorical with mode
    df['study_hours'] = df['study_hours'].fillna(df['study_hours'].mean())
    df['attendance_percentage'] = df['attendance_percentage'].fillna(df['attendance_percentage'].mean())
    df['previous_grades'] = df['previous_grades'].fillna(df['previous_grades'].mean())
    df['tutoring'] = df['tutoring'].fillna(df['tutoring'].mode()[0])

    # Encode categorical columns (Yes=1, No=0)
    df['tutoring'] = df['tutoring'].map({'Yes': 1, 'No': 0})

    # Define Features (X) and Target (y)
    X = df[['study_hours', 'attendance_percentage', 'previous_grades', 'tutoring']]
    y = df['passed']

    # 3. Split dataset
    print("Splitting dataset into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training samples: {len(X_train)} | Testing samples: {len(X_test)}")

    # 4. Train model
    print("Training Logistic Regression model...")
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # 5. Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")

    # 6. Interactive Prediction
    print("\n--- Make a Prediction ---")
    
    while True:
        try:
            print("\nEnter student details (or type 'exit' to quit):")
            
            hours_input = input("Study hours per week (e.g., 10): ")
            if hours_input.strip().lower() == 'exit':
                break
            hours = float(hours_input)
            
            attendance = float(input("Attendance percentage (0-100): "))
            grades = float(input("Previous grades (0-100): "))
            
            tutor_input = input("Has tutoring? (yes/no): ").strip().lower()
            tutoring = 1 if tutor_input == 'yes' else 0
            
            user_data = pd.DataFrame({
                'study_hours': [hours],
                'attendance_percentage': [attendance],
                'previous_grades': [grades],
                'tutoring': [tutoring]
            })
            
            prediction = model.predict(user_data)
            
            print("\n----- Result -----")
            if prediction[0] == 1:
                print("Prediction: The student is likely to PASS.")
            else:
                print("Prediction: The student is likely to FAIL.")
            print("------------------")
            
        except ValueError:
            print("Invalid input. Please enter numbers for hours, attendance, and grades.")

if __name__ == "__main__":
    main()
