from fastapi import FastAPI
from pydantic import BaseModel
import google.generativeai as genai
from dotenv import load_dotenv
import os
from report_schema import manual_schema_dict
from PIL import Image
from utilities.model_evaluation import generate_confidence_score
from utilities.pdf_byte_content import fetch_pdf

load_dotenv()
GEMINI_KEY=os.getenv("GEMINI_KEY")

class ConfidenceScoreRequestBody(BaseModel):
    pdf_url:str

app = FastAPI()
genai.configure(api_key=GEMINI_KEY)

@app.get("/")
def read_root():
    return {"message": "Hello from FastAPI!"}

@app.post("/confidence-score")
async def confidence_score_estimate(data:ConfidenceScoreRequestBody):
    model=genai.GenerativeModel(model_name="gemini-1.5-flash")
    tools = [{
            "function_declarations": [{
                "name": "parse_medical_report_details",
                "description": "Returns medical report values after parsing the uploaded image",
                "parameters":manual_schema_dict
            }]
    }]
    # image_path="./sample.png"
    # try:
    #     print(f"Attempting to open image: {image_path}")
    #     img = Image.open(image_path)
    #     print("Image opened successfully.")
    # except FileNotFoundError:
    #     print(f"Error: Image file not found at {image_path}", file=sys.stderr)
    #     return None
    # except Exception as e:
    #     print(f"Error opening or reading image file {image_path}: {e}", file=sys.stderr)
    #     return None

    # pdf_path="./modified_medical_report2.pdf"
    try:
        # with open(pdf_path, "rb") as f: # Open in binary read mode
        #     pdf_data = f.read()
        # if not pdf_data:
        #      print(f"Error: PDF file is empty at {pdf_path}")
        #      return {"success": False, "error": f"PDF file is empty at {pdf_path}"}

        # print(f"PDF file read successfully ({len(pdf_data)} bytes).")

        byte_content=await fetch_pdf(data.pdf_url)
        pdf_file_data = {
            "mime_type": "application/pdf",
            "data": byte_content
        }
    except Exception as e:
        print(f"Error opening or reading PDF file : {e}")
        return {"success": False, "response_text": f"Error reading PDF file: {e}"}
    try:
        response = await model.generate_content_async(["""
        You are a pdf reviewing agent who parses medical records. Review the uploaded pdf, it is a sample medical report.
        I want you to match the fields in this medical report to what is expected in the parameters of the parse_medical_report_details function
        to the best of your knowledge by matching commonly used acronyms, abbreviations for the medical terms involved and show some flexibility in term matching but never assume or approximate values. 
        For missing fields, I want you to make them null after you have verified that you can't possibly find them. If you don't find the uploaded pdf to be a medical report, ask the user to submit one.
        For boolean values, either convert from integer or infer from 'Yes' and 'No'"""
            ,pdf_file_data],
            tools=tools,
            tool_config={
            "function_calling_config": { 
            "mode": "auto",
            }},
            generation_config={
                "temperature":0.08
            }
        )
        if response.parts:
            part = response.parts[0]
            if part.function_call:            
                function_args = {}
                missing_fileds=[]
                for key, value in part.function_call.args.items():
                    if(value==None):
                        missing_fileds.append(key)
                    function_args[key]=value
                
                if(len(missing_fileds)>0):
                    return {
                        "success":False,
                        "response_text":"Missing fields",
                        "missing_fields":missing_fileds
                    }

                print(f"Arguments extracted: {function_args}")

                gender_map = {"male": 2, "female": 1}
                cardiac_list = [
                    function_args["age"]*365,
                    gender_map.get(function_args["gender"], 0),
                    function_args["height"],
                    function_args["weight"],
                    function_args["systolic_bp"],
                    function_args["diastolic_bp"],
                    function_args["cholesterol_level"],
                    function_args["glucose"],
                    int(function_args["smoke"]),
                    int(function_args["alcohol"]),
                    int(function_args["active"]),
                ]
                # chronic_list = [
                #     function_args["blood_pressure"],
                #     function_args["specific_gravity"],
                #     function_args["albumin"],
                #     int(function_args["sugar"]),
                #     int(function_args["rbc_in_urine"]),
                #     function_args["blood_urea"],
                #     function_args["serum_creatinine"],
                #     function_args["sodium"],
                #     function_args["potassium"],
                #     function_args["hemoglobin"],
                #     function_args["wbc_count"],
                #     function_args["rbc_count"],
                #     int(function_args["hypertension"]),
                # ]
                gender_map2 = {'female': 0, 'male': 1, 'other': 2}
                smoking_map = {'never': 0, 'former': 1, 'current': 2, 'ever': 3, 'not current': 4}
                bmi = function_args["weight"] / ((function_args["height"]/100) ** 2)

                diabetes_list = [
                    function_args["age"],
                    int(function_args["hypertension"]),
                    int(function_args["heart_disease"]),
                    bmi,
                    function_args["HbA1c_level"],
                    function_args["glucose"],
                    gender_map2.get(function_args["gender"], 0),
                    smoking_map.get(function_args["smoking_history"], 0)
                ]
                ls=generate_confidence_score(diabetes_list,cardiac_list)
                print(ls)
                return {
                    "success": True,
                    "output":ls
                }
            elif part.text:
                print(f"Received text response instead of function call: {part.text}")
                return {
                    "success": True, 
                    "response_text": part.text
                }
            else:
                print("Warning: Response part contained neither text nor function call.")
                return {"success": False, "response_text": "Sorry, invalid request formation at the server"}
        else:
            print("Error: No content parts received in the Gemini API response.")
            return {"success": False, "response_text": "Request failure, please try again"}
    except Exception as e:
        print(f"Error occurred in connecting with Gemini: {e}")
        return {"success":False,"response_text":"Internal Server Error"}


@app.get("/sample")
async def sample_pass():
    sample_chro=[70.0, 1.005, 4.0, 0.0, 1.0, 56.0, 3.8, 111.0, 2.5, 11.2, 6700.0, 3.9, 1.0]
    sample_card=[20228.0, 1.0, 156.0, 85.0, 140.0, 90.0, 3.0, 1.0, 0.0, 0.0, 1.0]
    ls=generate_confidence_score(sample_chro,sample_card)
    print(ls)
    return {
        "output":ls
    }
