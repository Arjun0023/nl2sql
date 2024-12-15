from fastapi import FastAPI, HTTPException, status, UploadFile, File
from pydantic import BaseModel
import openai
import sqlite3
import pandas as pd
from dotenv import load_dotenv
import os
import re
from typing import List, Dict, Any
import tempfile
from datetime import datetime
from fastapi.encoders import jsonable_encoder
import logging

logging.basicConfig(level=logging.INFO)

load_dotenv()
together_api_key = os.getenv("TOGETHER_API_KEY")
if not together_api_key:
    raise ValueError("TOGETHER_API_KEY not found in .env file")

client = openai.OpenAI(
    api_key=together_api_key,
    base_url="https://api.together.xyz/v1"
)
# Create database folder if it doesn't exist
DATABASE_DIR = "databases"
os.makedirs(DATABASE_DIR, exist_ok=True)

app = FastAPI()

def get_db_connection(db_path: str):
    try:
        conn = sqlite3.connect(db_path)
        return conn
    except sqlite3.Error as e:
        raise HTTPException(status_code=500, detail=f"Database connection error: {e}")

def get_table_names(db_path: str):
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    conn.close()
    return [table[0] for table in tables]


def create_table_from_df(conn: sqlite3.Connection, df: pd.DataFrame, table_name: str):
    try:
      #sanitize column names for SQL
       def sanitize_column_name(col):
            # remove all invalid characters, and if column starts with a number, replace it with an underscore.
            sanitized_col = re.sub(r'[^a-zA-Z0-9_]', '_', col).lower()
            if re.match(r'^[0-9]', sanitized_col):
              sanitized_col = f"_{sanitized_col}"
            if not sanitized_col:
              sanitized_col = "_unnamed" # handle empty column names
            return sanitized_col
       sanitized_columns = [sanitize_column_name(col) for col in df.columns]
       logging.info(f"Sanitized Columns: {list(zip(df.columns, sanitized_columns))}")
       df.columns = sanitized_columns
       df.to_sql(table_name, conn, if_exists="replace", index=False) #Replace table if it exists
    except Exception as e:
        raise Exception(f"Error creating table: {e}")
    return sanitized_columns
def process_uploaded_file(file: UploadFile) -> str:
  temp_db_path = None
  try:
    logging.info(f"Uploaded file name: {file.filename}, file size: {file.size}")
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(file.file.read())
        tmp_file_path = tmp_file.name
    
    if file.filename.lower().endswith(('.csv')):
          df = pd.read_csv(tmp_file_path)
    elif file.filename.lower().endswith(('.xlsx', '.xls')):
        xls = pd.ExcelFile(tmp_file_path) # use ExcelFile to handle older versions
        df = xls.parse(xls.sheet_names[0])  #read the first sheet
    else :
      raise Exception("Invalid file format")
    
    if df.empty:
        raise Exception("Dataframe is empty")
    logging.info(f"Dataframe columns: {df.columns.tolist()}")
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    db_filename = f"uploaded_data_{timestamp}.db" #Unique filename
    db_path = os.path.join(DATABASE_DIR, db_filename) # database path
    conn = get_db_connection(db_path)
    temp_db_path = db_path # set for cleanup
    table_name = "data" #table is always data for upload.
    sanitized_columns = create_table_from_df(conn, df, table_name)
    conn.close()
    os.unlink(tmp_file_path)
    return db_path, table_name #returning database file path, and table name.
  except Exception as e:
     if os.path.exists(tmp_file_path):
       os.unlink(tmp_file_path)
     if temp_db_path and os.path.exists(temp_db_path):
        os.remove(temp_db_path)
     raise Exception(f"Error processing file: {e}")
  


class QueryRequest(BaseModel):
    user_query: str
    component_type: str = "table" #defaulting to table if no component_type is specified
    db_path: str = None # new field for specifying database file path
    table_name: str = None # new field for specifying table


def is_valid_sql(query):
    sql_keywords = r'\b(SELECT|FROM|WHERE|JOIN|GROUP BY|ORDER BY|LIMIT)\b'
    return bool(re.search(sql_keywords, query, re.IGNORECASE))

def generate_sql(user_query:str, db_path : str, table_name: str = None):
    table_names = [table_name] if table_name else get_table_names(db_path) #if table is not specified, use all tables
    completion = client.chat.completions.create(
        model="meta-llama/Llama-3-8b-chat-hf",
        messages=[
            {"role": "system", "content": f"You are a helpful SQL assistant. Generate only the SQL query without any explanations or markdown formatting. The available tables are: {', '.join(table_names)}"},
            {"role": "user", "content": f"Generate SQL query for: {user_query}"}
        ]
    )
    sql_query = completion.choices[0].message.content.strip()
    return sql_query

def get_query_data(sql_query: str, db_path :str):
    if not is_valid_sql(sql_query):
        raise HTTPException(status_code=400, detail="The generated query doesn't appear to be valid SQL. Please try rephrasing your question.")
    conn = get_db_connection(db_path)
    try:
        cursor = conn.execute(sql_query)
        rows = cursor.fetchall()
        columns = [description[0] for description in cursor.description]
        conn.close()
        df = pd.DataFrame(rows, columns=columns)
        return df
    except sqlite3.Error as e:
        conn.close()
        raise HTTPException(status_code=500, detail=f"SQLite Error: {e}")

def generate_text_answer(sql_query: str, df: pd.DataFrame):
    completion = client.chat.completions.create(
        model="meta-llama/Llama-3-8b-chat-hf",
        messages=[
            {"role": "system", "content": "You are a helpful SQL assistant. Provide a concise, human-readable answer based on the query results."},
            {"role": "user", "content": f"Generate human-readable answer for SQL query: {sql_query}\nQuery result:\n{df.to_string()}"}
        ]
    )
    text_answer = completion.choices[0].message.content
    return text_answer

class ChartData(BaseModel):
  labels: List[str]
  datasets: List[Dict[str,Any]]

class CardData(BaseModel):
  value: str

class QueryResponse(BaseModel):
    text_answer: str
    component_type : str
    data: Any  #generic data response object


def get_chart_data(df: pd.DataFrame, chart_type: str) -> Dict[str,Any]: # return dict
  if chart_type == "Barchart":
    if len(df.columns) != 2:
      raise HTTPException(status_code=400, detail="Barchart requires exactly 2 columns in the result")
    x_col, y_col = df.columns[0], df.columns[1]
    labels = df[x_col].astype(str).tolist()
    datasets = [ { "label":y_col , "data":df[y_col].tolist() } ]
    return ChartData(labels=labels, datasets=datasets).model_dump()
  elif chart_type == "Piechart":
    if len(df.columns) != 2:
      raise HTTPException(status_code=400, detail="Piechart requires exactly 2 columns in the result")
    x_col, y_col = df.columns[0], df.columns[1]
    labels = df[x_col].astype(str).tolist()
    datasets = [{ "data":df[y_col].tolist() , "backgroundColor" : ["#FF6384","#36A2EB","#FFCE56","#4BC0C0","#9966FF"] }] # colors added for visual diversity
    return ChartData(labels=labels, datasets=datasets).model_dump()
  else :
    raise HTTPException(status_code=400, detail="Chart type not supported, Use Barchart , Piechart or table")


def get_card_data(df: pd.DataFrame) -> Dict[str,Any]:
    if len(df.columns) != 1:
      raise HTTPException(status_code=400, detail="Card component requires a single column from result")
    
    value = str(df.iloc[0, 0]) if not df.empty else "No Data"
    return CardData(value=value).model_dump()


def get_table_data(df: pd.DataFrame) -> List[Dict[str, Any]]:
    return df.to_dict(orient="records")


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        db_path, table_name = process_uploaded_file(file)
        return {"message": "File uploaded and processed successfully.", "db_path": db_path, "table_name" : table_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) #return just detail message, not full error


@app.post("/query", response_model=QueryResponse)
async def run_query(query_request: QueryRequest):
    try:
        sql_query = generate_sql(query_request.user_query, query_request.db_path , query_request.table_name)
        df = get_query_data(sql_query, query_request.db_path)
        text_answer = generate_text_answer(sql_query, df)
       
        component_type = query_request.component_type
        if component_type == "table":
          data =  get_table_data(df)
        elif component_type in ["Barchart", "Piechart"]:
            data = get_chart_data(df, component_type)
        elif component_type == "Card":
            data = get_card_data(df)
        else:
            raise HTTPException(status_code=400, detail="Invalid component type requested")
        
        return QueryResponse(
            text_answer=text_answer,
            component_type=component_type,
            data=data
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {e}")