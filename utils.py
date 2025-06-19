import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv(override=True)
google_api_key = os.getenv('GOOGLE_API_KEY')
if not google_api_key:
    raise ValueError("GOOGLE_API_KEY not found in environment variables")

genai.configure(api_key=google_api_key)
model = genai.GenerativeModel('gemini-pro')

def llm_invoke(prompt: str) -> str:
    response = model.generate_content(prompt)
    if hasattr(response, "text") and response.text:
        return response.text.strip()
    try:
        return response.candidates[0].content.parts[0].text.strip()
    except Exception:
        return ""

def get_csv_schema(df):
    return "\n".join([f"Column: {col} ({str(dtype)})" for col, dtype in df.dtypes.items()])

def get_excel_schema(sheets: dict) -> str:
    schema = []
    for sheet_name, df in sheets.items():
        cols = ', '.join([f"{col} ({str(dtype)})" for col, dtype in df.dtypes.items()])
        schema.append(f"Sheet: {sheet_name} | Columns: {cols}")
    return "\n".join(schema)