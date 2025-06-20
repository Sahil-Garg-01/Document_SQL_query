from fastapi import FastAPI, UploadFile, File, Form, Body
from pydantic import BaseModel
from fastapi.responses import StreamingResponse, JSONResponse
from mysql_module import get_workflow as get_mysql_workflow
import pandas as pd
import io
import os
import json

from db_postgres import get_workflow as get_pg_workflow, db_config
from csv_module import get_workflow as get_csv_workflow
from excel_module import get_workflow as get_excel_workflow
from utils import get_csv_schema, get_excel_schema

class UserInput(BaseModel):
    user_input: str

app = FastAPI()

app_graph = get_pg_workflow()
csv_app_graph = get_csv_workflow()
excel_app_graph = get_excel_workflow()
mysql_app_graph = get_mysql_workflow()

@app.post("/ask_postgres")
async def ask_question(payload: UserInput):
    user_input = payload.user_input
    if not db_config:
        return JSONResponse(
            status_code=400,
            content={"warning": "No database URL provided. Please upload a CSV or Excel file using /ask_csv or /ask_excel endpoint."}
        )
    result = app_graph.invoke({"user_input": user_input})
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

@app.post("/ask_csv")
async def ask_csv(user_input: str = Form(...), file: UploadFile = File(...)):
    # Parse user_input as JSON
    user_input_dict = json.loads(user_input)
    user_input_value = user_input_dict["user_input"]
    if not file.filename.endswith('.csv'):
        return JSONResponse(
            status_code=400,
            content={"error": "Only CSV files are allowed"}
        )
    df = pd.read_csv(file.file)
    schema = get_csv_schema(df)
    result = csv_app_graph.invoke({"user_input": user_input_value, "csv_schema": schema, "df": df})
    df_result = pd.DataFrame(result["results"])
    if df_result.empty:
        return JSONResponse(
            status_code=200,
            content={"message": "No results found for your query."}
        )
    stream = io.StringIO()
    df_result.to_csv(stream, index=False)
    stream.seek(0)
    return StreamingResponse(
        stream,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=results.csv"}
    )


@app.post("/ask_excel")
async def ask_excel(user_input: str = Form(...), file: UploadFile = File(...)):
    user_input_dict = json.loads(user_input)
    user_input_value = user_input_dict["user_input"]
    filename = file.filename.lower()
    if filename.endswith('.xlsx'):
        engine = "openpyxl"
    elif filename.endswith('.xls'):
        engine = "xlrd"
    else:
        return JSONResponse(
            status_code=400,
            content={"error": "Only Excel files (.xlsx, .xls) are allowed"}
        )
    sheets = pd.read_excel(file.file, sheet_name=None, engine=engine)
    schema = get_excel_schema(sheets)
    result = excel_app_graph.invoke({"user_input": user_input_value, "excel_schema": schema, "sheets": sheets})
    df_result = pd.DataFrame(result["results"])
    if df_result.empty:
        return JSONResponse(
            status_code=200,
            content={"message": "No results found for your query."}
        )
    stream = io.StringIO()
    df_result.to_csv(stream, index=False)
    stream.seek(0)
    return StreamingResponse(
        stream,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=results.csv"}
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