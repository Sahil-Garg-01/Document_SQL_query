from langgraph.graph import StateGraph, END, START
from pydantic import BaseModel
from typing import List, Dict
import pandas as pd
from utils import llm_invoke, get_excel_schema
from fastapi.responses import JSONResponse

class AgentState(BaseModel):
    user_input: str
    sql_query: str = ""
    results: List[Dict] = []
    excel_schema: str
    sheets: dict

    model_config = {"arbitrary_types_allowed": True}

def generate_excel_code(state: AgentState) -> AgentState:
    schema = state.excel_schema
    prompt = f"""
    Given the following Excel file schema (multiple sheets possible):
    {schema}

    Convert the following request to a single-line pandas DataFrame operation in Python that returns a DataFrame.
    Return ONLY the code, no explanations, no comments, no markdown, no print statements, and do NOT create a new DataFrame.
    Assume the Excel sheets are loaded as a dictionary of DataFrames called 'sheets', where keys are sheet names.
    Request: '{state.user_input}'
    """
    code = llm_invoke(prompt)
    return state.copy(update={"sql_query": code})

def execute_excel_code(state: AgentState) -> AgentState:
    sheets = state.sheets
    code = state.sql_query.strip()
    # Clean code fences if present
    if code.startswith("```"):
        code = code.lstrip("`")
        code = code.replace("python", "", 1).strip()
        code = code.rstrip("`").strip()
    try:
        result = eval(code, {"sheets": sheets, "pd": pd})
        if isinstance(result, pd.DataFrame):
            print("Excel results:", result)
            return state.copy(update={"results": result.to_dict(orient="records")})
        else:
            return state.copy(update={"results": [{"result": str(result)}]})
    except Exception as e:
        return state.copy(update={"results": [{"error": str(e)}]})

def get_workflow():
    workflow = StateGraph(AgentState)
    workflow.add_node("generate_code", generate_excel_code)
    workflow.add_node("execute_code", execute_excel_code)
    workflow.add_edge(START, "generate_code")
    workflow.add_edge("generate_code", "execute_code")
    workflow.add_edge("execute_code", END)
    return workflow.compile()

def handle_results(results):
    if not results or not isinstance(results, list):
        return JSONResponse(
            status_code=500,
            content={"error": "No valid results returned from Excel workflow.", "details": str(results)}
        )
    df_result = pd.DataFrame(results)
    if df_result.empty:
        return JSONResponse(
            status_code=200,
            content={"message": "No results found for your query."}
        )