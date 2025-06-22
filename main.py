from fastapi import FastAPI, UploadFile, File, Form, Body
from pydantic import BaseModel
from fastapi.responses import StreamingResponse, JSONResponse
from mysql_module import get_workflow as get_mysql_workflow
import pandas as pd
import io
import os
import json

from db_postgres import PostgresQueryAgent
from csv_module import CSVQueryAgent
from excel_module import ExcelQueryAgent
from utils import get_csv_schema, get_excel_schema

class UserInput(BaseModel):
    user_input: str

app = FastAPI()

app_graph = PostgresQueryAgent().get_workflow()
csv_app_graph = CSVQueryAgent().get_workflow()
excel_app_graph = ExcelQueryAgent().get_workflow()
mysql_app_graph = get_mysql_workflow()



@app.post("/ask_postgres")
async def ask_postgres(payload: UserInput):
    """
    Accepts a user query for the PostgreSQL database, runs the query using the Postgres agent,
    and returns the results as a CSV file.
    """
    try:
        user_input = payload.user_input
        agent = PostgresQueryAgent()
        if not agent.db_config:
            return JSONResponse(
                status_code=400,
                content={"warning": "No database URL provided. Please upload a CSV or Excel file using /ask_csv or /ask_excel endpoint."}
            )

        # Run the Postgres agent workflow
        workflow = agent.get_workflow()
        result = workflow.invoke({"user_input": user_input})

        # Prepare the result DataFrame
        df_result = pd.DataFrame(result["results"])
        if df_result.empty:
            return JSONResponse(
                status_code=200,
                content={"message": "No results found for your query."}
            )

        # Stream the results as a downloadable CSV file
        stream = io.StringIO()
        df_result.to_csv(stream, index=False)
        stream.seek(0)
        return StreamingResponse(
            stream,
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=results.csv"}
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Internal server error: {str(e)}"}
        )



@app.post("/ask_csv")
async def ask_csv(user_input: str = Form(...), file: UploadFile = File(...)):
    """
    Accepts a user query and a CSV file upload, dynamically generates the schema,
    runs the query using the CSV agent, and returns the results as a CSV file.
    """
    try:
        # Parse user_input as JSON and extract the actual query string
        user_input_dict = json.loads(user_input)
        user_input_value = user_input_dict.get("user_input", "")
        if not user_input_value:
            return JSONResponse(
                status_code=400,
                content={"error": "Missing 'user_input' in request."}
            )

        # Validate file type
        if not file.filename.lower().endswith('.csv'):
            return JSONResponse(
                status_code=400,
                content={"error": "Only CSV files are allowed."}
            )

        # Read the uploaded CSV file into a DataFrame
        df = pd.read_csv(file.file)

        # Dynamically generate the schema string from the DataFrame
        schema = get_csv_schema(df)

        # Run the CSV agent workflow
        result = csv_app_graph.invoke({
            "user_input": user_input_value,
            "csv_schema": schema,
            "df": df
        })

        # Prepare the result DataFrame
        df_result = pd.DataFrame(result["results"])
        if df_result.empty:
            return JSONResponse(
                status_code=200,
                content={"message": "No results found for your query."}
            )

        # Stream the results as a downloadable CSV file
        stream = io.StringIO()
        df_result.to_csv(stream, index=False)
        stream.seek(0)
        return StreamingResponse(
            stream,
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=results.csv"}
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Internal server error: {str(e)}"}
        )






@app.post("/ask_excel")
async def ask_excel(user_input: str = Form(...), file: UploadFile = File(...)):
    """
    Accepts a user query and an Excel file upload, dynamically generates the schema,
    runs the query using the Excel agent, and returns the results as a CSV file.
    """
    try:
        # Parse user_input as JSON and extract the actual query string
        user_input_dict = json.loads(user_input)
        user_input_value = user_input_dict.get("user_input", "")
        if not user_input_value:
            return JSONResponse(
                status_code=400,
                content={"error": "Missing 'user_input' in request."}
            )

        # Validate file type
        filename = file.filename.lower()
        if not (filename.endswith('.xlsx') or filename.endswith('.xls')):
            return JSONResponse(
                status_code=400,
                content={"error": "Only Excel files (.xlsx, .xls) are allowed."}
            )
        engine = "openpyxl" if filename.endswith('.xlsx') else "xlrd"

        # Read the uploaded Excel file into a dict of DataFrames
        sheets = pd.read_excel(file.file, sheet_name=None, engine=engine)

        # Dynamically generate the schema string from the sheets
        schema = get_excel_schema(sheets)

        # Run the Excel agent workflow
        result = excel_app_graph.invoke({
            "user_input": user_input_value,
            "excel_schema": schema,
            "sheets": sheets
        })

        # Prepare the result DataFrame
        df_result = pd.DataFrame(result["results"])
        if df_result.empty:
            return JSONResponse(
                status_code=200,
                content={"message": "No results found for your query."}
            )

        # Stream the results as a downloadable CSV file
        stream = io.StringIO()
        df_result.to_csv(stream, index=False)
        stream.seek(0)
        return StreamingResponse(
            stream,
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=results.csv"}
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Internal server error: {str(e)}"}
        )




@app.post("/ask_mysql")
async def ask_mysql(payload: UserInput):
    user_input = payload.user_input
    from mysql_module import db_config as mysql_db_config
    if not mysql_db_config:
        return JSONResponse(
            status_code=400,
            content={"warning": "No MySQL URL provided. Please set MYSQL_URL in your .env file."}
        )
    result = mysql_app_graph.invoke({"user_input": user_input})
    df = pd.DataFrame(result["results"])
    if df.empty:
        return JSONResponse(
            status_code=200,
            content={"message": "No results found for your query."}
        )
    stream = io.StringIO()
    df.to_csv(stream, index=False)
    stream.seek(0)
    return StreamingResponse(
        stream,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=results.csv"}
    )