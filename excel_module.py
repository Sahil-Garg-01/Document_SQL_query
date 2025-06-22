from langgraph.graph import StateGraph, END, START
from pydantic import BaseModel
from typing import List, Dict, Optional
import pandas as pd
from utils import llm_invoke, get_excel_schema

class AgentState(BaseModel):
    user_input: str
    sql_query: Optional[str] = None
    results: List[Dict] = []
    excel_schema: str
    sheets: dict

    model_config = {"arbitrary_types_allowed": True}

class ExcelQueryAgent:
    """
    Agent for generating and executing pandas code on Excel files using LLM.
    """

    def __init__(self):
        pass

    def generate_excel_code(self, state: AgentState) -> AgentState:
        schema = state.excel_schema
        # Extract sheet names for prompt clarity
        sheet_names = []
        for line in schema.splitlines():
            if ":" in line:
                sheet_names.append(line.split(":")[0].strip())
        if len(sheet_names) == 1:
            sheet_hint = f"The main sheet is called '{sheet_names[0]}'."
        else:
            sheet_hint = f"The sheets are: {', '.join([f"'{s}'" for s in sheet_names])}."

        prompt = f"""
Given the following Excel file schema:
{schema}

{sheet_hint}

Write valid Python code using pandas that answers the following request.
- Use only the sheets and columns shown above.
- Use ONLY the sheet names and columns exactly as shown above. Do NOT invent or guess sheet names or column names.
- If a requested sheet or column does not exist, return an empty DataFrame assigned to result.
- Assume the Excel sheets are loaded as a dictionary of DataFrames called 'sheets', where keys are sheet names.
- Select only the relevant columns needed to answer the request (avoid selecting all columns).
- Use correct sheet and column names as per the schema.
- If a merge (join) is needed, use the correct keys.
- If columns with the same name exist in both DataFrames, rename columns before merging to ensure uniqueness.
- When merging DataFrames, always specify the 'suffixes' parameter with unique values (e.g., suffixes=('_left', '_right')) to avoid duplicate column names.
- If aggregation, grouping, or filtering is needed, do so as per the request.
- Output valid, executable Python code (assignments and multi-line code allowed, but no print statements).
- Assign the final DataFrame to a variable named result.
- Return ONLY the code, nothing else.

Request: '{state.user_input}'
"""
        try:
            code = llm_invoke(prompt)
            code = code.strip()
            if code.startswith("```"):
                code = code.lstrip("`").replace("python", "", 1).strip().rstrip("`").strip()
            state.sql_query = code
            return state
        except Exception as e:
            state.results = [{"error": f"Code generation failed: {str(e)}"}]
            return state

    def execute_excel_code(self, state: AgentState) -> AgentState:
        sheets = state.sheets
        code = (state.sql_query or "").strip()
        if code.startswith("```"):
            code = code.lstrip("`").replace("python", "", 1).strip().rstrip("`").strip()
        local_vars = {"sheets": sheets, "pd": pd}
        try:
            exec(code, {}, local_vars)
            result = local_vars.get("result", None)
            if isinstance(result, pd.DataFrame):
                state.results = result.to_dict(orient="records")
            elif result is not None:
                state.results = [{"result": str(result)}]
            else:
                state.results = [{"error": "No result DataFrame produced by code."}]
        except SyntaxError as se:
            state.results = [{"error": f"Syntax error in generated code: {str(se)}", "code": code}]
        except Exception as e:
            state.results = [{"error": f"Execution error: {str(e)}", "code": code}]
        return state

    def get_workflow(self):
        """Define the workflow for the agent."""
        workflow = StateGraph(AgentState)
        workflow.add_node("generate_code", self.generate_excel_code)
        workflow.add_node("execute_code", self.execute_excel_code)
        workflow.add_edge(START, "generate_code")
        workflow.add_edge("generate_code", "execute_code")
        workflow.add_edge("execute_code", END)
        return workflow.compile()

# Example usage
def main(user_input: str, excel_schema: str, sheets: dict):
    agent = ExcelQueryAgent()
    state = AgentState(
        user_input=user_input,
        excel_schema=excel_schema,
        sheets=sheets
    )
    workflow = agent.get_workflow()
    final_state = workflow.run(state)
    print(final_state.results)

if __name__ == "__main__":
    excel_path = input("Enter the path to your Excel file: ")
    # Load all sheets as DataFrames
    sheets = pd.read_excel(excel_path, sheet_name=None)
    excel_schema = get_excel_schema(sheets)
    user_input = input("Enter your query: ")

    # print("Excel sheets:", list(sheets.keys()))
    # for name, df in sheets.items():
    #     print(f"First rows of {name}:\n", df.head())
    # print("Extracted schema:", excel_schema)

    main(user_input, excel_schema, sheets)