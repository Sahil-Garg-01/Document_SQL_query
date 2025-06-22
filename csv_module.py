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

class CSVQueryAgent:
    """
    Agent for generating and executing pandas code on denormalized CSVs using LLM.
    """

    def __init__(self):
        pass

    def generate_pandas_code(self, state: AgentState) -> AgentState:
        """
        Generates pandas code using LLM based on the user input and CSV schema.
        """
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
- Assign the final DataFrame to a variable named result.
- Return ONLY the code, nothing else.
- When converting columns to datetime, do so only in the relevant DataFrame (e.g., after splitting orders, use orders['order_date'] = pd.to_datetime(orders['order_date'], errors='coerce')).
- Always use errors='coerce' with pd.to_datetime to avoid parsing errors.

Request: '{state.user_input}'
"""
        try:
            code = llm_invoke(prompt)
            code = code.strip()
            if code.startswith("```"):
                code = code.lstrip("`").replace("python", "", 1).strip().rstrip("`").strip()
            state.sql_query = code
            # print("Generated code:", state.sql_query)
            return state
        except Exception as e:
            state.results = [{"error": f"Code generation failed: {str(e)}"}]
            return state

    def execute_pandas_code(self, state: AgentState) -> AgentState:
        """
        Executes the generated pandas code safely and updates the state with results or errors.
        """
        df = state.df
        code = (state.sql_query or "").strip()
        if code.startswith("```"):
            code = code.lstrip("`").replace("python", "", 1).strip().rstrip("`").strip()
        local_vars = {"df": df, "pd": pd}
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
        workflow.add_node("generate_code", self.generate_pandas_code)
        workflow.add_node("execute_code", self.execute_pandas_code)
        workflow.add_edge(START, "generate_code")
        workflow.add_edge("generate_code", "execute_code")
        workflow.add_edge("execute_code", END)
        return workflow.compile()

# Example usage
def main(user_input: str, csv_schema: str, df: pd.DataFrame):
    """
    Runs the CSVQueryAgent workflow for any user input, schema, and DataFrame.
    """
    agent = CSVQueryAgent()
    state = AgentState(
        user_input=user_input,
        csv_schema=csv_schema,
        df=df
    )
    workflow = agent.get_workflow()
    final_state = workflow.run(state)
    print(final_state.results)

if __name__ == "__main__":
   
    csv_path = input("Enter the path to your CSV file: ")
    df = pd.read_csv(csv_path)
    csv_schema = get_csv_schema(df)
    user_input = input("Enter your query: ")

    # print("CSV columns:", df.columns)
    # print("First rows:\n", df.head())
    # print("Extracted schema:", csv_schema)

    main(user_input, csv_schema, df)