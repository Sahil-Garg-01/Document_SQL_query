from langgraph.graph import StateGraph, END, START
from pydantic import BaseModel
from typing import List, Dict, Optional
import pandas as pd
from utils import llm_invoke, get_csv_schema

class AgentState(BaseModel):
    user_input: str
    sql_query: Optional[str] = None
    results: List[Dict] = []
    csv_schema: str  
    df: pd.DataFrame

    model_config = {"arbitrary_types_allowed": True}

def generate_pandas_code(state: AgentState) -> AgentState:
    schema = state.csv_schema
    prompt = f"""
    Given the following CSV schema:
    {schema}

    Write valid Python code using pandas that answers the following request.
    - The CSV contains all tables in one file, with a 'table_name' column indicating the type of row.
    - First, split the DataFrame into separate DataFrames for each table using df[df['table_name'] == 'table'] and selecting only the relevant columns for each table as shown in the schema.
    - Use only the columns shown above. Do NOT invent or guess column names.
    - If the user asks for a column that does not exist in the schema, return an empty DataFrame (e.g., pd.DataFrame()).
    - Always include in the output all columns explicitly requested by the user, if they exist in the schema.
    - Assume the DataFrame is already loaded as 'df'.
    - Select only the relevant columns needed to answer the request (avoid selecting all columns).
    - If aggregation, grouping, or filtering is needed, do so as per the request.
    - When filtering on columns that may contain missing values (e.g., rating), use .isna() or .fillna() to avoid errors.
    - When writing query strings, use single quotes inside the string (e.g., .query('year == 2025')).
    - Output valid, executable Python code (assignments and multi-line code allowed, but no print statements).
    - Return ONLY the code, nothing else.
    - When converting columns to datetime, do so only in the relevant DataFrame (e.g., after splitting orders, use orders['order_date'] = pd.to_datetime(orders['order_date'], errors='coerce')).
    - Always use errors='coerce' with pd.to_datetime to avoid parsing errors.

    Request: '{state.user_input}'
    """
    code = llm_invoke(prompt)
    return state.copy(update={"sql_query": code})

def execute_pandas_code(state: AgentState) -> AgentState:
    df = state.df
    code = state.sql_query.strip()
    # Remove code fences if present
    if code.startswith("```"):
        code = code.lstrip("`")
        code = code.replace("python", "", 1).strip()
        code = code.rstrip("`").strip()
    try:
        # Prepare a local namespace for exec
        local_vars = {"df": df, "pd": pd}
        # The LLM should assign the final result to a variable named 'result'
        exec(code, {}, local_vars)
        result = local_vars.get("result", None)
        if isinstance(result, pd.DataFrame):
            return state.copy(update={"results": result.to_dict(orient="records")})
        else:
            return state.copy(update={"results": [{"result": str(result)}]})
    except Exception as e:
        return state.copy(update={"results": [{"error": str(e)}]})

def get_workflow():
    workflow = StateGraph(AgentState)
    workflow.add_node("generate_code", generate_pandas_code)
    workflow.add_node("execute_code", execute_pandas_code)
    workflow.add_edge(START, "generate_code")
    workflow.add_edge("generate_code", "execute_code")
    workflow.add_edge("execute_code", END)
    return workflow.compile()


