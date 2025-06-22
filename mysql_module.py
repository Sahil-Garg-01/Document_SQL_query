from langgraph.graph import StateGraph, END, START
from pydantic import BaseModel
from typing import List, Dict, Optional
import mysql.connector
import os
from urllib.parse import urlparse
from utils import llm_invoke
from dotenv import load_dotenv

load_dotenv()

class AgentState(BaseModel):
    user_input: str
    sql_query: Optional[str] = ""
    results: List[Dict] = []

    model_config = {"arbitrary_types_allowed": True}

class MySQLQueryAgent:
    """
    Agent for generating and executing SQL queries on a MySQL database using LLM.
    """

    def __init__(self):
        db_url = os.getenv('MYSQL_URL')
        if db_url:
            url = urlparse(db_url)
            self.db_config = {
                "host": url.hostname,
                "port": url.port or 3306,
                "user": url.username,
                "password": url.password,
                "database": url.path.lstrip("/")
            }
        else:
            self.db_config = None

    def get_db_conn(self):
        if not self.db_config:
            raise RuntimeError("MySQL URL not provided. Please upload a CSV or Excel file instead.")
        return mysql.connector.connect(**self.db_config)

    def get_schema(self) -> str:
        if not self.db_config:
            raise RuntimeError("MySQL URL not provided. Please upload a CSV or Excel file instead.")
        conn = self.get_db_conn()
        cursor = conn.cursor()
        try:
            cursor.execute("""
                SELECT TABLE_NAME, COLUMN_NAME
                FROM information_schema.COLUMNS
                WHERE TABLE_SCHEMA = %s
            """, (self.db_config["database"],))
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

    def generate_sql(self, state: AgentState) -> AgentState:
        schema = self.get_schema()
        # Extract table names for prompt clarity
        table_names = []
        for line in schema.splitlines():
            if ":" in line and not line.startswith("Tables and columns"):
                table_names.append(line.split(":")[0].strip())
        if len(table_names) == 1:
            table_hint = f"The main table is called '{table_names[0]}'."
        else:
            table_hint = f"The tables are: {', '.join([f"'{t}'" for t in table_names])}."

        prompt = f"""
Given the following MySQL database schema:
{schema}

{table_hint}

Convert the following natural language request to a valid SQL query for this schema.
Return ONLY the SQL query, no explanations, no comments, no markdown, no code fences, and no language tags.
- Use only the tables and columns shown above.
- Select only the relevant columns needed to answer the request (avoid SELECT *).
- Use correct table and column names as per the schema.
- Use case-insensitive matching (e.g., lower(column) = lower('value')) for text comparisons.
- If a JOIN is needed, use the correct keys.
- If the request asks for a person's email, assume the name is in a column like 'name', 'username', or 'full_name'.
- If no exact column match, return an empty query and note the issue.
- If aggregation, grouping, or filtering is needed, do so as per the request.
- Return ONLY the SQL query, no explanations, no comments, no markdown, no code fences, and no language tags.

Request: '{state.user_input}'
"""
        try:
            sql_query = llm_invoke(prompt)
            sql_query = sql_query.strip()
            if sql_query.startswith("```"):
                sql_query = sql_query.lstrip("`").replace("sql", "", 1).strip().rstrip("`").strip()
            state.sql_query = sql_query
            return state
        except Exception as e:
            state.results = [{"error": f"SQL generation failed: {str(e)}"}]
            return state

    def execute_query(self, state: AgentState) -> AgentState:
        sql = (state.sql_query or "").strip()
        if sql.startswith("```"):
            sql = sql.lstrip("`").replace("sql", "", 1).strip().rstrip("`").strip()
        try:
            conn = self.get_db_conn()
            cursor = conn.cursor(dictionary=True)
            try:
                cursor.execute(sql)
                results = cursor.fetchall()
                state.results = results
            except Exception as e:
                state.results = [{"error": f"SQL execution failed: {str(e)}", "query": sql}]
            finally:
                cursor.close()
                conn.close()
        except Exception as e:
            state.results = [{"error": f"Database connection failed: {str(e)}"}]
        return state

    def get_workflow(self):
        """Define the workflow for the agent."""
        workflow = StateGraph(AgentState)
        workflow.add_node("generate_sql", self.generate_sql)
        workflow.add_node("execute_query", self.execute_query)
        workflow.add_edge(START, "generate_sql")
        workflow.add_edge("generate_sql", "execute_query")
        workflow.add_edge("execute_query", END)
        return workflow.compile()

# Example usage
def main(user_input: str):
    agent = MySQLQueryAgent()
    state = AgentState(user_input=user_input)
    workflow = agent.get_workflow()
    final_state = workflow.run(state)
    print(final_state.results)

if __name__ == "__main__":
    user_input = input("Enter your query: ")
    main(user_input)