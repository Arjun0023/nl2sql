import os
import json
import numpy as np
import pandas as pd
import tempfile
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from dotenv import load_dotenv
import google.generativeai as genai
import sqlite3  # Import sqlite3 for database operations
import re
from typing import List, Dict, Union  # For better type hinting

# Load environment variables
load_dotenv()

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# System Instruction for SQL Query Generation
SYSTEM_INSTRUCTION = """You are an expert SQL query assistant. You ONLY respond with valid SQL code that answers the user's question based on the provided table schema.
Do NOT include any explanations, markdown formatting, or surrounding text. Focus solely on generating the correct SQL. If you cannot create a SQL query, respond with an empty string. IMPORTANT: Don't include \n or any formatting in the response."""

# System Instruction for SQL Query Fixing
SQL_FIXER_INSTRUCTION = """You are an expert SQL query fixer. You are provided with a user question, a generated SQL query that produced an error, the error message, and the columns of the dataset.  Your task is to fix the SQL query so that it will execute without errors and accurately answer the user's question. You MUST return a JSON object with the following structure:
{
  "fixed_query": "The corrected SQL query",
  "explanation": "A brief explanation of the changes made and why they were necessary"
}
If you cannot fix the query, return {"fixed_query": null, "explanation": "Unable to fix the query."}"""


class SQLQueryRequest(BaseModel):
    question: str


def convert_numpy_types(obj):
    """
    Recursively convert NumPy types to native Python types
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Series):
        return obj.tolist()
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    return obj


class GeminiDataAnalyzer:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")

        genai.configure(api_key=api_key)

        self.generation_config = {
            "temperature": 0.3,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,
        }

        self.model = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            generation_config=self.generation_config,
            system_instruction=SYSTEM_INSTRUCTION
        )
        self.sql_fixer_model = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            generation_config=self.generation_config,
            system_instruction=SQL_FIXER_INSTRUCTION
        )

    def generate_sql_query(self, question: str, table_name: str, schema: str) -> str:
        """Generate SQL query for a given natural language question"""
        try:
            prompt = f"""
            Given the following SQL table, create a SQL query to answer the question.
            Table Name: {table_name}
            Table Schema: {schema}
            Question: {question}
            Respond ONLY with the SQL query, nothing else. If you can't create a SQL query, respond with an empty string.
            IMPORTANT: Don't include \n or any formatting in the response.
            """
            response = self.model.generate_content(prompt)
            sql_query = response.text.strip()
            sql_query = re.sub(r"```sql|```", "", sql_query).strip()  # Remove the ```sql and ``` from the response
            return sql_query
        except Exception as e:
            raise Exception(f"Error generating SQL query: {str(e)}")

    def analyze_dataframe(self, df: pd.DataFrame) -> dict:
        """Analyze the uploaded DataFrame and provide insights"""
        try:
            # Convert column types to strings for JSON serialization
            column_types = {col: str(dtype) for col, dtype in df.dtypes.items()}

            # Basic dataset information
            info = {
                "total_rows": len(df),
                "total_columns": len(df.columns),
                "columns": list(df.columns),
                "column_types": column_types,
                "summary_stats": {}
            }

            # Add summary statistics for numeric columns
            numeric_columns = df.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns
            for col in numeric_columns:
                info["summary_stats"][col] = {
                    "mean": convert_numpy_types(df[col].mean()),
                    "median": convert_numpy_types(df[col].median()),
                    "min": convert_numpy_types(df[col].min()),
                    "max": convert_numpy_types(df[col].max()),
                    "std": convert_numpy_types(df[col].std())
                }

            # Check for datetime columns
            datetime_columns = df.select_dtypes(include=['datetime64']).columns
            info["datetime_columns"] = list(datetime_columns)

            # Generate possible queries based on dataset
            query_prompt = f"""
            Based on this dataset with columns: {', '.join(df.columns)}
            Suggest 5 interesting SQL queries that could provide insights.
            """
            suggested_queries_response = self.model.generate_content(query_prompt)
            info["suggested_queries"] = suggested_queries_response.text.strip()

            return convert_numpy_types(info)
        except Exception as e:
            raise Exception(f"Error analyzing DataFrame: {str(e)}")

    def fix_sql_query(self, question: str, sql_query: str, error_message: str, columns: List[str]) -> Dict[str, str]:
        """Attempts to fix a broken SQL query using the LLM."""
        try:
            prompt = f"""
            User Question: {question}
            SQL Query: {sql_query}
            Error Message: {error_message}
            Dataset Columns: {', '.join(columns)}

            Fix the SQL query so that it executes without errors and accurately answers the user's question. Return a JSON object with "fixed_query" and "explanation" keys. If you cannot fix the query, return {"fixed_query": null, "explanation": "Unable to fix the query."}
            """
            response = self.sql_fixer_model.generate_content(prompt)
            try:
                result = json.loads(response.text)
                if not isinstance(result, dict) or "fixed_query" not in result or "explanation" not in result:
                    print(f"Unexpected response format from SQL fixer: {response.text}")  # Log the unexpected response
                    return {"fixed_query": None, "explanation": "Unable to fix the query due to an unexpected response from the fixer."}
                return result
            except json.JSONDecodeError as e:
                print(f"JSONDecodeError: {e}, Response: {response.text}")  # Log the error and response
                return {"fixed_query": None, "explanation": "Unable to fix the query due to a JSON parsing error."}


        except Exception as e:
            print(f"Error in fix_sql_query: {str(e)}")
            return {"fixed_query": None, "explanation": f"Unable to fix the query due to an error: {str(e)}"}


# Initialize Gemini Data Analyzer
data_analyzer = GeminiDataAnalyzer()

# Store the uploaded DataFrame and its filename
uploaded_df = None
uploaded_filename = None


def create_chart_data(result: List[Dict[str, Union[int, float, str]]]) -> Dict[str, List[Union[int, float, str]]]:
    """
    Transforms the SQL query result into a format suitable for charting.
    Assumes the result has at least two columns: one for labels (e.g., category names)
    and one for values (e.g., counts or sums).  Modify as needed based on your data.

    Args:
        result: The list of dictionaries returned from the SQL query.

    Returns:
        A dictionary with 'labels' and 'values' keys, each containing a list.
    """
    if not result:
        return {"labels": [], "values": []}

    # Assuming first column is labels and the second column is values.
    labels = [str(row[list(row.keys())[0]]) for row in result]
    values = [row[list(row.keys())[1]] for row in result]  # Handle different numeric types

    return {"labels": labels, "values": values}


@app.post("/query")
async def generate_query(request: SQLQueryRequest):
    global uploaded_df, uploaded_filename

    if uploaded_df is None:
        raise HTTPException(status_code=400, detail="No file has been uploaded yet.")

    try:
        # Generate SQL query
        table_name = os.path.splitext(uploaded_filename)[0]  # Use filename as table name
        schema = ", ".join([f"{col} {str(uploaded_df[col].dtype)}" for col in
                             uploaded_df.columns])  # Create schema description
        sql_query = data_analyzer.generate_sql_query(request.question, table_name, schema)
        sql_query = sql_query.replace('\n', ' ').strip()
        print(sql_query)
        if not sql_query:
            raise HTTPException(status_code=400, detail="LLM could not generate a valid SQL query.")

        # Execute the SQL query on the DataFrame
        conn = sqlite3.connect(":memory:")  # In-memory database
        uploaded_df.to_sql(table_name, conn, if_exists="replace", index=False)

        try:
            result = pd.read_sql_query(sql_query, conn)
            result_json = jsonable_encoder(convert_numpy_types(result.to_dict(orient="records")))  # Convert to JSON
            chart_data = create_chart_data(result.to_dict(orient="records"))  # Create data for chart
        except Exception as e:
            error_message = str(e)
            print(f"SQL Execution Error: {error_message}")
            conn.close()  # Close connection before calling the fixer

            # Attempt to fix the SQL query
            fix_result = data_analyzer.fix_sql_query(
                request.question,
                sql_query,
                error_message,
                list(uploaded_df.columns)
            )

            if fix_result and fix_result["fixed_query"]:
                fixed_sql_query = fix_result["fixed_query"]
                print(f"Fixed SQL Query: {fixed_sql_query}")

                # Retry executing the fixed query
                conn = sqlite3.connect(":memory:")
                uploaded_df.to_sql(table_name, conn, if_exists="replace", index=False)  # Reload DF
                try:
                    result = pd.read_sql_query(fixed_sql_query, conn)
                    result_json = jsonable_encoder(convert_numpy_types(result.to_dict(orient="records")))
                    chart_data = create_chart_data(result.to_dict(orient="records"))
                    conn.close()
                    return {
                        "status": "success",
                        "query": fixed_sql_query,
                        "result": result_json,
                        "chartData": chart_data,
                        "explanation": fix_result["explanation"]
                    }
                except Exception as e:
                    conn.close()
                    raise HTTPException(
                        status_code=400,
                        detail=f"Fixed query still failed: {str(e)}.  Original query: {sql_query}, Fixed Query: {fixed_sql_query}, Explanation: {fix_result['explanation']}"
                    )
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Error executing SQL query: {error_message}. Check your query or ensure it aligns with the data structure. SQL Query: {sql_query}.  Unable to automatically fix."
                ) # Propagate the original exception

        finally:
            conn.close()

        return {
            "status": "success",
            "query": sql_query,
            "result": result_json,
            "chartData": chart_data  # Include the chart data in the response
        }

    except HTTPException as http_exc:  # Catch already handled HTTPExceptions
        raise http_exc
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    global uploaded_df, uploaded_filename
    try:
        # Validate file type
        allowed_extensions = ['.csv', '.xlsx', '.xls']
        file_ext = os.path.splitext(file.filename)[1].lower()

        if file_ext not in allowed_extensions:
            raise HTTPException(status_code=400, detail="Unsupported file type. Please upload CSV or Excel files.")

        # Create a temporary file to save the uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            temp_file.write(await file.read())
            temp_file_path = temp_file.name

        try:
            # Read the file based on its extension
            if file_ext == '.csv':
                uploaded_df = pd.read_csv(temp_file_path)
            else:  # xlsx or xls
                uploaded_df = pd.read_excel(temp_file_path)

            uploaded_filename = file.filename  # Store the filename

            # Analyze the DataFrame
            analysis_result = data_analyzer.analyze_dataframe(uploaded_df)

            # Optional: Clean up the temporary file
            os.unlink(temp_file_path)

            # Use jsonable_encoder to ensure JSON serialization
            return {
                "status": "success",
                "filename": file.filename,
                "analysis": jsonable_encoder(analysis_result)
            }

        except Exception as e:
            # Ensure temp file is deleted even if processing fails
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
            raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)