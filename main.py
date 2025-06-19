from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse, JSONResponse
from mysql_module import get_workflow as get_mysql_workflow
import pandas as pd
import io

from db_postgres import get_workflow as get_pg_workflow, db_config
from csv_module import get_workflow as get_csv_workflow
from excel_module import get_workflow as get_excel_workflow
from utils import get_csv_schema, get_excel_schema

app = FastAPI()

app_graph = get_pg_workflow()
csv_app_graph = get_csv_workflow()
excel_app_graph = get_excel_workflow()
mysql_app_graph = get_mysql_workflow()

@app.post("/ask_postgres")
async def ask_question(user_input: str = Form(...)):
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
    if not file.filename.endswith('.csv'):
        return JSONResponse(
            status_code=400,
            content={"error": "Only CSV files are allowed"}
        )
    df = pd.read_csv(file.file)
    schema = get_csv_schema(df)
    result = csv_app_graph.invoke({"user_input": user_input, "csv_schema": schema, "df": df})
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
    if not (file.filename.endswith('.xlsx') or file.filename.endswith('.xls')):
        return JSONResponse(
            status_code=400,
            content={"error": "Only Excel files (.xlsx, .xls) are allowed"}
        )
    sheets = pd.read_excel(file.file, sheet_name=None)
    schema = get_excel_schema(sheets)
    result = excel_app_graph.invoke({"user_input": user_input, "excel_schema": schema, "sheets": sheets})
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
async def ask_mysql(user_input: str = Form(...)):
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