manual_schema_dict = {
    "type": "OBJECT", 
    "properties": {
        "age":{
            "type":"INTEGER",
            "description":"The patient's age provided in the report"
        },
        "gender":{
            "type":"STRING",
            "enum":["female","male","other"],
            "description":"The gender of the patient"
        },
        "blood_pressure": {
            "type": "INTEGER", # Use "INTEGER" if using direct protos
            "description": "The patient's blood pressure reading."
        },
        "height": {
            "type": "INTEGER", 
            "description": "The height in cm of the patient."
        },
        "weight": {
            "type": "NUMBER", 
            "description": "The height in kg of the patient."
        },
        "hypertension": {
            "type": "BOOLEAN", 
            "description": "Whether the patient has hypertension"
        },
        "heart_disease": {
            "type": "BOOLEAN", 
            "description": "Whether the patient has a heart disease or not listed in the report"
        },
        "HbA1c_level":{
            "type":"NUMBER",
            "description":"The HbA1c_level of the patient in the report"
        },
        "smoking_history":{
            "type":"STRING",
            "enum":['current', 'ever', 'former', 'never', 'not current'],
            "description":"The smoking history of the patient specifically mentioned in the report"
        },
         "systolic_bp": {"type": "INTEGER", "description": "Systolic blood pressure"},
         "diastolic_bp": {"type": "INTEGER", "description": "Diastolic blood pressure"},
         "cholesterol_level": {"type": "INTEGER", "description": "Cholesterol level"},
         "glucose": {"type": "INTEGER", "description": "Glucose level"},
         "smoke": {"type": "BOOLEAN", "description": "Smoking status"},
         "alcohol": {"type": "BOOLEAN", "description": "Alcohol consumption status"},
         "active": {"type": "BOOLEAN", "description": "Activity status to be converted to boolean if given in integer"}
    },
    "required": [
        'blood_pressure',"smoking_history","HbA1c_level","heart_disease",
         'hypertension', 'systolic_bp', 'diastolic_bp',
        'cholesterol_level', 'glucose', 'smoke', 'alcohol', 'active','gender','age','height','weight'
         ]
}