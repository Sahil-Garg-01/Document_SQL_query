from langgraph.graph import StateGraph, END, START
from pydantic import BaseModel
from typing import List, Dict
import psycopg2
import psycopg2.extras
import os
from urllib.parse import urlparse
from utils import llm_invoke

from dotenv import load_dotenv
load_dotenv()

class AgentState(BaseModel):
    user_input: str
    sql_query: str = ""
    results: List[Dict] = []

db_url = os.getenv('DB_URL')
# print("DB_URL from .env:", db_url)
if db_url:
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
            SELECT table_name, column_name
            FROM information_schema.columns
            WHERE table_schema = 'public'
        """)
        schema = {}
        for table, column in cursor.fetchall():
            schema.setdefault(table, []).append(column)
        schema_lines = ["Tables and columns:"]
        for table, columns in schema.items():
            schema_lines.append(f"{table}: {', '.join(columns)}")
        return "\n".join(schema_lines)
    finally:
        cursor.close()
        conn.close()

def generate_sql(state: AgentState) -> AgentState:
    schema = get_schema()
    # Dynamically extract all table names from the schema string
    table_names = []
    for line in schema.splitlines():
        if line.startswith("Table:"):
            table_names.append(line.split("Table:")[1].strip())
    # If only one table, use it as the main table; else, list all
    if len(table_names) == 1:
        table_hint = f"The main table is called '{table_names[0]}'."
    else:
        table_hint = f"The tables are: {', '.join([f"'{t}'" for t in table_names])}."

    prompt = f"""
    Given the following PostgreSQL database schema:
    {schema}

    {table_hint}

    Convert the following natural language request to a valid SQL query for this schema.
    Return ONLY the SQL query, no explanations, no comments, no markdown, no code fences, and no language tags.
    - Use only the tables and columns shown above.
    - Select only the relevant columns needed to answer the request (avoid SELECT *) c.
    - Use correct table and column names as per the schema.
    - Use case-insensitive matching (e.g., lower(column) = lower('value')) for text comparisons.
    - If a JOIN is needed, use the correct keys.
    - If the request asks for a person's email, assume the name is in a column like 'name', 'username', or 'full_name'.
    - If no exact column match, return an empty query and note the issue.
    - If aggregation, grouping, or filtering is needed, do so as per the request.
    - Return ONLY the SQL query, no explanations, no comments, no markdown, no code fences, and no language tags.

    Request: '{state.user_input}'
    """
    sql_query = llm_invoke(prompt)
    # print("Generated SQL:", sql_query)
    return state.copy(update={"sql_query": sql_query})

def execute_query(state: AgentState) -> AgentState:
    conn = get_db_conn()
    cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    sql = state.sql_query.strip()
    # Remove code fences if present
    if sql.startswith("```"):
        sql = sql.lstrip("`")
        sql = sql.replace("sql", "", 1).strip()
        sql = sql.rstrip("`").strip()
    try:
        cursor.execute(sql)
        results = cursor.fetchall()
    except Exception as e:
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




# user_input: "Find the top 5 users who spent the most in 2025, with their names, emails, total spent, number of orders, and average rating given (for ratings 4 or higher)"
'''sql_query = SELECT 
    u.id AS user_id,
    u.name,
    u.email,
    COUNT(o.id) AS order_count,
    SUM(p.price) AS total_spent,
    ROUND(AVG(r.rating)::NUMERIC, 2) AS avg_rating_given
FROM 
    users u
    INNER JOIN orders o ON u.id = o.user_id
    INNER JOIN products p ON o.product_id = p.id
    LEFT JOIN reviews r ON u.id = r.user_id AND p.id = r.product_id
WHERE 
    o.order_date BETWEEN '2025-01-01' AND '2025-12-31'
    AND (r.rating IS NULL OR r.rating >= 4)
GROUP BY 
    u.id, u.name, u.email
HAVING 
    COUNT(o.id) >= 1
ORDER BY 
    total_spent DESC
LIMIT 5;'''