from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict
import mysql.connector
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI   
from urllib.parse import urlparse
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# State definition for LangGraph
class AgentState(TypedDict):
    user_input: str
    sql_query: str
    results: List[Dict]

# Get environment variables
db_url = os.getenv('DB_URL')
if not db_url:
    raise ValueError("DB_URL not found in environment variables")

llm_api_key = os.getenv('LLM_API_KEY')
if not llm_api_key:
    raise ValueError("LLM_API_KEY not found in environment variables")

# LLM setup
llm = ChatOpenAI(api_key=llm_api_key)

# MySQL connection from URL
url = urlparse(db_url)
db_config = {
    "host": url.hostname,
    "port": url.port or 3306,
    "user": url.username,
    "password": url.password,
    "database": url.path.lstrip("/")
}

# Fetch schema dynamically
def get_schema() -> str:
    conn = mysql.connector.connect(**db_config)
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

# Step 1: Generate SQL query using LLM
def generate_sql(state: AgentState) -> AgentState:
    schema = get_schema()
    prompt = PromptTemplate(
        input_variables=["schema", "user_input"],
        template="""
        Given the database schema:
        {schema}
        
        Convert the following request to a SQL query. Select only the relevant columns needed to answer the request. Avoid using SELECT *. Return only the SQL query.
        Request: '{user_input}'
        """
    )
    sql_query = llm.invoke(prompt.format(schema=schema, user_input=state["user_input"])).strip()
    return {"sql_query": sql_query}

# Step 2: Execute SQL query
def execute_query(state: AgentState) -> AgentState:
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute(state["sql_query"])
        results = cursor.fetchall()
    except mysql.connector.Error as e:
        results = [{"error": str(e)}]
    finally:
        cursor.close()
        conn.close()
    return {"results": results}

# LangGraph workflow
workflow = StateGraph(AgentState)
workflow.add_node("generate_sql", generate_sql)
workflow.add_node("execute_query", execute_query)
workflow.add_edge("generate_sql", "execute_query")
workflow.add_edge("execute_query", END)
app = workflow.compile()

# Run agent
def run_agent(user_input: str) -> List[Dict]:
    result = app.invoke({"user_input": user_input})
    return result["results"]

# Example usage
if __name__ == "__main__":
    inputs = ["Show employee names in Sales department", "Get highest salary with employee name"]
    for input_text in inputs:
        print(f"Input: {input_text}")
        results = run_agent(input_text)
        print("Results:", results)