import streamlit as st
import openai
import sqlite3
import pandas as pd
from dotenv import load_dotenv
import os
import re

load_dotenv()
together_api_key = os.getenv("TOGETHER_API_KEY")

client = openai.OpenAI(
    api_key=together_api_key,
    base_url="https://api.together.xyz/v1"
)

# Use an absolute path or ensure the relative path is correct
db_path = './RevenueData.db'
st.write(f"Database path: {os.path.abspath(db_path)}")

try:
    conn = sqlite3.connect(db_path)
    st.success("Successfully connected to the database.")
    
    # Fetch and display all table names
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    st.write("Tables in the database:", [table[0] for table in tables])
except sqlite3.Error as e:
    st.error(f"Error connecting to database: {e}")
    st.stop()

st.title("SQL Agent Demo")

user_query = st.text_input("Enter your query", placeholder="Query goes here")

def is_valid_sql(query):
    # Basic SQL validation
    sql_keywords = r'\b(SELECT|FROM|WHERE|JOIN|GROUP BY|ORDER BY|LIMIT)\b'
    return bool(re.search(sql_keywords, query, re.IGNORECASE))

if st.button("Run Query"):
    try:
        completion = client.chat.completions.create(
            model="meta-llama/Llama-3-8b-chat-hf",
            messages=[
                {"role": "system", "content": f"You are a helpful SQL assistant. Generate only the SQL query without any explanations or markdown formatting. The available tables are: {', '.join([table[0] for table in tables])}"},
                {"role": "user", "content": f"Generate SQL query for: {user_query}"}
            ]
        )
        sql_query = completion.choices[0].message.content.strip()

        st.write("Generated SQL Query:")
        st.code(sql_query, language="sql")

        if not is_valid_sql(sql_query):
            st.error("The generated query doesn't appear to be valid SQL. Please try rephrasing your question.")
        else:
            cursor = conn.execute(sql_query)
            rows = cursor.fetchall()
            columns = [description[0] for description in cursor.description]

            df = pd.DataFrame(rows, columns=columns)

            completion = client.chat.completions.create(
                model="meta-llama/Llama-3-8b-chat-hf",
                messages=[
                    {"role": "system", "content": "You are a helpful SQL assistant. Provide a concise, human-readable answer based on the query results."},
                    {"role": "user", "content": f"Generate human-readable answer for SQL query: {sql_query}\nQuery result:\n{df.to_string()}"}
                ]
            )
            text_answer = completion.choices[0].message.content

            st.write("Text Answer:")
            st.markdown(text_answer)
            st.write("Table Output:")
            st.table(df)

    except sqlite3.Error as e:
        st.error(f"SQLite Error: {e}")
        st.write("Error details:")
        st.write(f"SQL Query: {sql_query}")
        st.write(f"Available tables: {[table[0] for table in tables]}")
    except Exception as e:
        st.error(f"Error: {e}")

conn.close()
