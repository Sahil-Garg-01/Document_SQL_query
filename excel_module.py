from langgraph.graph import StateGraph, END, START
from pydantic import BaseModel
from typing import List, Dict
import pandas as pd
from utils import llm_invoke, get_excel_schema

class AgentState(BaseModel):
    user_input: str
    sql_query: str = ""
    results: List[Dict] = []
    schema: str
    sheets: dict

def generate_excel_code(state: AgentState) -> AgentState:
    schema = state["schema"]
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
    return {**state, "sql_query": code}

def execute_excel_code(state: AgentState) -> AgentState:
    sheets = state["sheets"]
    try:
        result = eval(state["sql_query"], {"sheets": sheets, "pd": pd})
        if isinstance(result, pd.DataFrame):
            return {**state, "results": result.to_dict(orient="records")}
        else:
            return {**state, "results": [{"result": str(result)}]}
    except Exception as e:
        return {**state, "results": [{"error": str(e)}]}

def get_workflow():
    workflow = StateGraph(AgentState)
    workflow.add_node("generate_code", generate_excel_code)
    workflow.add_node("execute_code", execute_excel_code)
    workflow.add_edge(START, "generate_code")
    workflow.add_edge("generate_code", "execute_code")
    workflow.add_edge("execute_code", END)
    return workflow.compile()