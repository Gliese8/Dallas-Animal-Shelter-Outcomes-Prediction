import pandas as pd
import joblib
import sqlite3
from sklearn.preprocessing import StandardScaler

def classify_samples(model_path, input_file):
    # Load the model, feature columns, and scaler
    model = joblib.load(model_path)
    model_features = joblib.load('model_features_svm.pkl')
    scaler = joblib.load('scaler_svm.pkl')
    
    input_data = pd.read_csv(input_file)
    
    ids = input_data['id']
    
    input_data_encoded = pd.get_dummies(input_data, columns=['Animal_Type', 'Animal_Breed', 'Intake_Type', 'Intake_Subtype', 
                                                             'Reason', 'Intake_Condition', 'Hold_Request', 
                                                             'Outcome_Subtype', 'Outcome_Condition', 'Chip_Status', 
                                                             'Animal_Origin'])
    
    # Ensure the input data has the same columns as the training data
    input_data_encoded = input_data_encoded.reindex(columns=model_features, fill_value=0)
    
    # Handle any remaining NaN values
    input_data_encoded = input_data_encoded.fillna(0)
    
    input_data_encoded['Duration_of_Stay'] = scaler.transform(input_data_encoded[['Duration_of_Stay']])
    
    predictions = model.predict(input_data_encoded)
    
    result_df = pd.DataFrame({'id': ids, 'class': predictions})
    
    # Save results to SQLite database
    conn = sqlite3.connect('outcome_predictions.db')
    cursor = conn.cursor()
    
    # Check if table exists and create if it doesn't
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS outcome_predictions (
        id INTEGER PRIMARY KEY,
        class TEXT
    )
    """)
    conn.commit()

    # Insert new data or update existing data if ID already exists
    for _, row in result_df.iterrows():
        cursor.execute("SELECT 1 FROM outcome_predictions WHERE id = ?", (row['id'],))
        if cursor.fetchone():
            cursor.execute("UPDATE outcome_predictions SET class = ? WHERE id = ?", (row['class'], row['id']))
        else:
            cursor.execute("INSERT INTO outcome_predictions (id, class) VALUES (?, ?)", (row['id'], row['class']))
    
    conn.commit()
    conn.close()

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python classify_samples.py <model_path> <input_file>")
    else:
        model_path = sys.argv[1]
        input_file = sys.argv[2]
        classify_samples(model_path, input_file)
