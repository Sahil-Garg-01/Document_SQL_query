from langgraph.graph import StateGraph, END, START
from pydantic import BaseModel
from typing import List, Dict
import mysql.connector
import os
from urllib.parse import urlparse
from utils import llm_invoke

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
    
    Convert the following request to a SQL query. Select only the relevant columns needed to answer the request. Avoid using SELECT *. Return only the SQL query.
    Request: '{state["user_input"]}'
    """
    sql_query = llm_invoke(prompt)
    return {**state, "sql_query": sql_query}

def execute_query(state: AgentState) -> AgentState:
    conn = get_db_conn()
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute(state["sql_query"])
        results = cursor.fetchall()
    except mysql.connector.Error as e:
        results = [{"error": str(e)}]
    finally:
        cursor.close()
        conn.close()
    return {**state, "results": results}

def get_workflow():
    workflow = StateGraph(AgentState)
    workflow.add_node("generate_sql", generate_sql)
    workflow.add_node("execute_query", execute_query)
    workflow.add_edge(START, "generate_sql")
    workflow.add_edge("generate_sql", "execute_query")
    workflow.add_edge("execute_query", END)
    return workflow.compile()