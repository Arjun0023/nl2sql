import streamlit as st
import openai
import sqlite3
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from dotenv import load_dotenv
import os
import re

# Initialize session state variables
if 'sql_query' not in st.session_state:
    st.session_state.sql_query = ''
if 'df' not in st.session_state:
    st.session_state.df = None
if 'text_answer' not in st.session_state:
    st.session_state.text_answer = ''

load_dotenv()
together_api_key = os.getenv("TOGETHER_API_KEY")
client = openai.OpenAI(
    api_key=together_api_key,
    base_url="https://api.together.xyz/v1"
)

# Use an absolute path or ensure the relative path is correct
db_path = './TrainingData.db'
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

st.title("Final Year Project Prototype")

# Function to reset session state
def reset_session_state():
    st.session_state.sql_query = ''
    st.session_state.df = None
    st.session_state.text_answer = ''

def is_valid_sql(query):
    # Basic SQL validation
    sql_keywords = r'\b(SELECT|FROM|WHERE|JOIN|GROUP BY|ORDER BY|LIMIT)\b'
    return bool(re.search(sql_keywords, query, re.IGNORECASE))

def create_visualizations(df):
    """
    Create pie charts and bar charts based on the DataFrame
    """
    # Visualization container to prevent page reset
    with st.container():
        # Visualization options
        viz_type = st.radio("Choose Visualization Type", 
                            ["None", "Pie Chart", "Bar Chart", "Both"],
                            key='viz_type_select')
        
        if viz_type in ["Pie Chart", "Both"]:
            st.subheader("Pie Chart")
            # Try to find a categorical column and a numeric column
            cat_columns = df.select_dtypes(include=['object']).columns
            num_columns = df.select_dtypes(include=['int64', 'float64']).columns
            
            if len(cat_columns) > 0 and len(num_columns) > 0:
                # Let user choose columns
                pie_cat_col = st.selectbox("Choose Category Column for Pie Chart", 
                                           cat_columns, 
                                           key='pie_cat_column')
                pie_num_col = st.selectbox("Choose Numeric Column for Pie Chart", 
                                           num_columns, 
                                           key='pie_num_column')
                
                # Create pie chart
                pie_fig = px.pie(df, names=pie_cat_col, values=pie_num_col, 
                                 title=f"{pie_num_col} by {pie_cat_col}")
                st.plotly_chart(pie_fig, use_container_width=True)
        
        if viz_type in ["Bar Chart", "Both"]:
            st.subheader("Bar Chart")
            # Try to find a categorical column and a numeric column
            cat_columns = df.select_dtypes(include=['object']).columns
            num_columns = df.select_dtypes(include=['int64', 'float64']).columns
            
            if len(cat_columns) > 0 and len(num_columns) > 0:
                # Let user choose columns
                bar_cat_col = st.selectbox("Choose Category Column for Bar Chart", 
                                           cat_columns, 
                                           key='bar_cat_column')
                bar_num_col = st.selectbox("Choose Numeric Column for Bar Chart", 
                                           num_columns, 
                                           key='bar_num_column')
                
                # Create bar chart
                bar_fig = px.bar(df, x=bar_cat_col, y=bar_num_col, 
                                 title=f"{bar_num_col} by {bar_cat_col}")
                st.plotly_chart(bar_fig, use_container_width=True)

# Main query processing
def process_query(user_query):
    try:
        completion = client.chat.completions.create(
            model="meta-llama/Llama-3-8b-chat-hf",
            messages=[
                {"role": "system", "content": f"You are a helpful SQL assistant. Generate only the SQL query without any explanations or markdown formatting. The available tables are: {', '.join([table[0] for table in tables])}"},
                {"role": "user", "content": f"Generate SQL query for: {user_query}"}
            ]
        )
        sql_query = completion.choices[0].message.content.strip()
        st.session_state.sql_query = sql_query

        st.write("Generated SQL Query:")
        st.code(sql_query, language="sql")

        if not is_valid_sql(sql_query):
            st.error("The generated query doesn't appear to be valid SQL. Please try rephrasing your question.")
            return None

        cursor = conn.execute(sql_query)
        rows = cursor.fetchall()
        columns = [description[0] for description in cursor.description]

        df = pd.DataFrame(rows, columns=columns)
        st.session_state.df = df

        completion = client.chat.completions.create(
            model="meta-llama/Llama-3-8b-chat-hf",
            messages=[
                {"role": "system", "content": "You are a helpful SQL assistant. Provide a concise, human-readable answer based on the query results."},
                {"role": "user", "content": f"Generate human-readable answer for SQL query: {sql_query}\nQuery result:\n{df.to_string()}"}
            ]
        )
        text_answer = completion.choices[0].message.content
        st.session_state.text_answer = text_answer

        return df

    except sqlite3.Error as e:
        st.error(f"SQLite Error: {e}")
        st.write("Error details:")
        st.write(f"SQL Query: {sql_query}")
        st.write(f"Available tables: {[table[0] for table in tables]}")
    except Exception as e:
        st.error(f"Error: {e}")
    
    return None
def generate_summary_and_insights(df, user_query):
    if df is not None and not df.empty:
        try:
            completion = client.chat.completions.create(
                model="meta-llama/Llama-3-8b-chat-hf",
                messages=[
                    {"role": "system", "content": "You are a data analysis assistant. Provide a concise summary of the data and actionable insights. Focus on key trends, outliers, and any important information."},
                    {"role": "user", "content": f"User query: {user_query}\nData:\n{df.to_string(index=False)}"}
                ]
            )
            insights = completion.choices[0].message.content
            return insights
        except Exception as e:
            st.error(f"Error generating insights: {e}")
            return "Unable to generate insights due to an error."
    else:
        return "No data available for generating insights."

# Sidebar for query input
with st.sidebar:
    st.header("Query Input")
    user_query = st.text_input("Enter your query", placeholder="Query goes here", key='user_query_input')
    
    # Query submission button
    if st.button("Run Query", key='run_query_button'):
        # Reset session state when a new query is run
        reset_session_state()
        
        # Process the query
        result_df = process_query(user_query)

# Main content area
if st.session_state.df is not None:
    # st.write("Text Answer:")
    # st.markdown(st.session_state.text_answer)
    
    st.write("Table Output:")
    st.table(st.session_state.df)
    st.write("Summary and Actionable Insights:")
    insights = generate_summary_and_insights(st.session_state.df, user_query)
    st.markdown(insights)
    # Visualizations
    create_visualizations(st.session_state.df)

conn.close()