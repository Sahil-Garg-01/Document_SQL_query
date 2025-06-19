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
    Given the CSV schema:
    {schema}

    Convert the following request to a single-line pandas DataFrame operation in Python that returns a DataFrame.
    Return ONLY the code, no explanations, no imports, no function definitions, no markdown, no comments, no print statements, and do NOT create a new DataFrame.
    Assume the DataFrame is already loaded as 'df'.
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
        result = eval(code, {"df": df, "pd": pd})
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


