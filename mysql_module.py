from langgraph.graph import StateGraph, END, START
from pydantic import BaseModel
from typing import List, Dict
import mysql.connector
import os
from urllib.parse import urlparse
from utils import llm_invoke
from dotenv import load_dotenv
load_dotenv()

class AgentState(BaseModel):
    user_input: str
    sql_query: str = ""
    results: List[Dict] = []

db_url = os.getenv('MYSQL_URL')
if db_url:
    url = urlparse(db_url)
    db_config = {
        "host": url.hostname,
        "port": url.port or 3306,
        "user": url.username,
        "password": url.password,
        "database": url.path.lstrip("/")
    }
else:
    db_config = None

def get_db_conn():
    if not db_config:
        raise RuntimeError("MySQL URL not provided. Please upload a CSV or Excel file instead.")
    return mysql.connector.connect(**db_config)

def get_schema() -> str:
    if not db_config:
        raise RuntimeError("MySQL URL not provided. Please upload a CSV or Excel file instead.")
    conn = get_db_conn()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            SELECT TABLE_NAME, COLUMN_NAME, DATA_TYPE 
            FROM information_schema.COLUMNS 
            WHERE TABLE_SCHEMA = %s
        """, (db_config["database"],))
        schema = []
        for table, column, dtype in cursor.fetchall():
            schema.append(f"Table: {table}, Column: {column} ({dtype})")
        return "\n".join(schema)
    finally:
        cursor.close()
        conn.close()

def generate_sql(state: AgentState) -> AgentState:
    schema = get_schema()
    prompt = f"""
    Given the database schema:
    {schema}
    
    Convert the following request to a SQL query. Select only the relevant columns needed to answer the request. Avoid using SELECT *. 
    Return ONLY the SQL query, no explanations, no comments, no markdown, no code fences, and no language tags.
    Request: '{state.user_input}'
    """
    sql_query = llm_invoke(prompt)
    return state.copy(update={"sql_query": sql_query})

def execute_query(state: AgentState) -> AgentState:
    conn = get_db_conn()
    cursor = conn.cursor(dictionary=True)
    sql = state.sql_query.strip()
    # Remove code fences if present
    if sql.startswith("```"):
        sql = sql.lstrip("`")
        sql = sql.replace("sql", "", 1).strip()
        sql = sql.rstrip("`").strip()
    try:
        cursor.execute(sql)
        results = cursor.fetchall()
    except mysql.connector.Error as e:
        results = [{"error": str(e)}]
    finally:
        cursor.close()
        conn.close()
    return state.copy(update={"results": results})

def get_workflow():
    workflow = StateGraph(AgentState)
    workflow.add_node("generate_sql", generate_sql)
    workflow.add_node("execute_query", execute_query)
    workflow.add_edge(START, "generate_sql")
    workflow.add_edge("generate_sql", "execute_query")
    workflow.add_edge("execute_query", END)
    return workflow.compile()