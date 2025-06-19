from langgraph.graph import StateGraph, END, START
from pydantic import BaseModel
from typing import List, Dict
import pandas as pd
from utils import llm_invoke, get_csv_schema

class AgentState(BaseModel):
    user_input: str
    sql_query: str = "" 
    results: List[Dict] = []
    schema: str
    df: pd.DataFrame

def generate_pandas_code(state: AgentState) -> AgentState:
    schema = state["schema"]
    prompt = f"""
    Given the CSV schema:
    {schema}
    
    Convert the following request to a pandas DataFrame operation in Python. Return only the code to get the answer as a DataFrame.
    Request: '{state["user_input"]}'
    """
    code = llm_invoke(prompt)
    return {**state, "sql_query": code}

def execute_pandas_code(state: AgentState) -> AgentState:
    df = state["df"]
    try:
        result = eval(state["sql_query"], {"df": df, "pd": pd})
        if isinstance(result, pd.DataFrame):
            return {**state, "results": result.to_dict(orient="records")}
        else:
            return {**state, "results": [{"result": str(result)}]}
    except Exception as e:
        return {**state, "results": [{"error": str(e)}]}

def get_workflow():
    workflow = StateGraph(AgentState)
    workflow.add_node("generate_code", generate_pandas_code)
    workflow.add_node("execute_code", execute_pandas_code)
    workflow.add_edge(START, "generate_code")
    workflow.add_edge("generate_code", "execute_code")
    workflow.add_edge("execute_code", END)
    return workflow.compile()


