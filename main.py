from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse, JSONResponse
from typing import TypedDict, List, Dict
from langgraph.graph import StateGraph, END, START
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import pandas as pd
import psycopg2
import psycopg2.extras
import os
import io
import google.generativeai as genai

# Load environment variables
load_dotenv(override=True)

class AgentState(TypedDict):
    user_input: str
    sql_query: str
    results: List[Dict]

db_url = os.getenv('DB_URL')
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

if db_url:
    from urllib.parse import urlparse
    url = urlparse(db_url)
    db_config = {
        "host": url.hostname,
        "port": url.port or 5432,
        "user": url.username,
        "password": url.password,
        "dbname": url.path.lstrip("/")
    }
else:
    db_config = None

def get_db_conn():
    if not db_config:
        raise RuntimeError("Database URL not provided. Please upload a CSV or Excel file instead.")
    return psycopg2.connect(**db_config)

def get_schema() -> str:
    if not db_config:
        raise RuntimeError("Database URL not provided. Please upload a CSV or Excel file instead.")
    conn = get_db_conn()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            SELECT table_name, column_name, data_type
            FROM information_schema.columns
            WHERE table_schema = 'public'
        """)
        schema = []
        for table, column, dtype in cursor.fetchall():
            schema.append(f"Table: {table}, Column: {column} ({dtype})")
        return "\n".join(schema)
    finally:
        cursor.close()
        conn.close()

def generate_sql(state: AgentState, schema: str) -> AgentState:
    prompt = f"""
    Given the database schema:
    {schema}
    
    Convert the following request to a SQL query. Select only the relevant columns needed to answer the request. Avoid using SELECT *. Return only the SQL query.
    Request: '{state["user_input"]}'
    """
    sql_query = llm_invoke(prompt)
    return {"sql_query": sql_query}

def execute_query(state: AgentState) -> AgentState:
    conn = get_db_conn()
    cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    try:
        cursor.execute(state["sql_query"])
        results = cursor.fetchall()
    except Exception as e:
        results = [{"error": str(e)}]
    finally:
        cursor.close()
        conn.close()
    return {"results": results}

# LangGraph workflow for DB
workflow = StateGraph(AgentState)
workflow.add_node("generate_sql", lambda state: generate_sql(state, get_schema()))
workflow.add_node("execute_query", execute_query)
workflow.add_edge(START, "generate_sql")
workflow.add_edge("generate_sql", "execute_query")
workflow.add_edge("execute_query", END)
app_graph = workflow.compile()

# For CSV
def get_csv_schema(df: pd.DataFrame) -> str:
    return "\n".join([f"Column: {col} ({str(dtype)})" for col, dtype in df.dtypes.items()])

def generate_pandas_code(state: AgentState, schema: str) -> AgentState:
    prompt = f"""
    Given the CSV schema:
    {schema}
    
    Convert the following request to a pandas DataFrame operation in Python. Return only the code to get the answer as a DataFrame.
    Request: '{state["user_input"]}'
    """
    code = llm_invoke(prompt)
    return {"sql_query": code}

def execute_pandas_code(state: AgentState, df: pd.DataFrame) -> AgentState:
    # WARNING: Using eval on LLM output is dangerous. Sandbox in production!
    try:
        result = eval(state["sql_query"], {"df": df, "pd": pd})
        if isinstance(result, pd.DataFrame):
            return {"results": result.to_dict(orient="records")}
        else:
            return {"results": [{"result": str(result)}]}
    except Exception as e:
        return {"results": [{"error": str(e)}]}

csv_workflow = StateGraph(AgentState)
csv_workflow.add_node("generate_code", lambda state: generate_pandas_code(state, state["schema"]))
csv_workflow.add_node("execute_code", lambda state: execute_pandas_code(state, state["df"]))
csv_workflow.add_edge(START, "generate_code")
csv_workflow.add_edge("generate_code", "execute_code")
csv_workflow.add_edge("execute_code", END)
csv_app_graph = csv_workflow.compile()

# For Excel (multi-sheet)
def get_excel_schema(sheets: dict) -> str:
    schema = []
    for sheet_name, df in sheets.items():
        cols = ', '.join([f"{col} ({str(dtype)})" for col, dtype in df.dtypes.items()])
        schema.append(f"Sheet: {sheet_name} | Columns: {cols}")
    return "\n".join(schema)

def generate_excel_code(state: dict, schema: str) -> dict:
    prompt = f"""
    Given the following Excel file schema (multiple sheets possible):
    {schema}

    The user may refer to a specific sheet by name, or just ask a question.
    If not specified, infer the most relevant sheet.
    Write a pandas DataFrame operation in Python to answer the request.
    Use the variable 'sheets' (a dict of DataFrames, keys are sheet names).
    Return only the code to get the answer as a DataFrame.
    Request: '{state["user_input"]}'
    """
    code = llm_invoke(prompt)
    return {"sql_query": code}

def execute_excel_code(state: dict, sheets: dict) -> dict:
    # WARNING: Using eval on LLM output is dangerous. Sandbox in production!
    try:
        result = eval(state["sql_query"], {"sheets": sheets, "pd": pd})
        if isinstance(result, pd.DataFrame):
            return {"results": result.to_dict(orient="records")}
        else:
            return {"results": [{"result": str(result)}]}
    except Exception as e:
        return {"results": [{"error": str(e)}]}

excel_workflow = StateGraph(dict)
excel_workflow.add_node("generate_code", lambda state: generate_excel_code(state, state["schema"]))
excel_workflow.add_node("execute_code", lambda state: execute_excel_code(state, state["sheets"]))
excel_workflow.add_edge(START, "generate_code")
excel_workflow.add_edge("generate_code", "execute_code")
excel_workflow.add_edge("execute_code", END)
excel_app_graph = excel_workflow.compile()

app = FastAPI()

@app.post("/ask")
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
    result = csv_app_graph.invoke({"user_input": user_input, "schema": schema, "df": df})
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
    result = excel_app_graph.invoke({"user_input": user_input, "schema": schema, "sheets": sheets})
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