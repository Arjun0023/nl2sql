import streamlit as st
import openai
import sqlite3
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from dotenv import load_dotenv
import os
import re
from io import BytesIO

# Initialize session state variables
if 'sql_query' not in st.session_state:
    st.session_state.sql_query = ''
if 'df' not in st.session_state:
    st.session_state.df = None
if 'text_answer' not in st.session_state:
    st.session_state.text_answer = ''
if 'table_name' not in st.session_state:
    st.session_state.table_name = None
if 'conn' not in st.session_state:
    st.session_state.conn = None

load_dotenv()
together_api_key = os.getenv("TOGETHER_API_KEY")
client = openai.OpenAI(
    api_key=together_api_key,
    base_url="https://api.together.xyz/v1"
)

def create_connection():
    """Create an in-memory SQLite database connection"""
    return sqlite3.connect(':memory:')

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

def generate_summary_and_insights(df, user_query):
    """Generate summary and insights from the data"""
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

def upload_and_process_file():
    """Handle file upload and create SQLite table"""
    uploaded_file = st.file_uploader("Upload your Excel or CSV file", type=['xlsx', 'csv'])
    
    if uploaded_file is not None:
        try:
            # Read the file
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            # Create a new database connection
            st.session_state.conn = create_connection()
            
            # Generate a table name from the file name
            table_name = re.sub(r'[^a-zA-Z0-9]', '_', uploaded_file.name.split('.')[0].lower())
            st.session_state.table_name = table_name
            
            # Save DataFrame to SQLite
            df.to_sql(table_name, st.session_state.conn, if_exists='replace', index=False)
            
            # Display success message and data preview
            st.success(f"File uploaded successfully! Table name: {table_name}")
            st.write("Data Preview:")
            st.write(df.head())
            #
            # Display column information
            st.write("Column Information:")
            column_info = pd.DataFrame({
                'Column Name': df.columns,
                'Data Type': df.dtypes.values
            })
            st.write(column_info)
            
            return df.columns.tolist()
            
        except Exception as e:
            st.error(f"Error processing file: {e}")
            return None
    return None

def generate_sql_prompt(columns, user_query, table_name):
    """Generate a prompt for the LLM to create SQL query"""
    column_info = "\n".join([f"- {col}" for col in columns])
    prompt = f"""Given a table named '{table_name}' with the following columns:
{column_info}

Generate a SQL query to answer: {user_query}

Remember to:
1. Use only the columns that exist in the table
2. Keep the query simple and efficient
3. Return relevant columns for visualization if appropriate

Generate ONLY the SQL query without any explanation or additional text."""
    return prompt

def is_valid_sql(query):
    """Basic SQL validation"""
    sql_keywords = r'\b(SELECT|FROM|WHERE|GROUP BY|ORDER BY|LIMIT)\b'
    return bool(re.search(sql_keywords, query, re.IGNORECASE))

def process_query(user_query, columns):
    """Process the user query and generate SQL"""
    if st.session_state.table_name is None:
        st.error("Please upload a file first!")
        return None
        
    try:
        # Generate SQL query using LLM
        prompt = generate_sql_prompt(columns, user_query, st.session_state.table_name)
        completion = client.chat.completions.create(
            model="meta-llama/Llama-3-8b-chat-hf",
            messages=[
                {"role": "system", "content": "You are a helpful SQL assistant. Generate only the SQL query without any explanations or markdown formatting."},
                {"role": "user", "content": prompt}
            ]
        )
        sql_query = completion.choices[0].message.content.strip()
        st.session_state.sql_query = sql_query

        st.write("Generated SQL Query:")
        st.code(sql_query, language="sql")

        if not is_valid_sql(sql_query):
            st.error("The generated query doesn't appear to be valid SQL. Please try rephrasing your question.")
            return None

        # Execute query
        cursor = st.session_state.conn.execute(sql_query)
        rows = cursor.fetchall()
        columns = [description[0] for description in cursor.description]

        df = pd.DataFrame(rows, columns=columns)
        st.session_state.df = df

        return df

    except sqlite3.Error as e:
        st.error(f"SQLite Error: {e}")
    except Exception as e:
        st.error(f"Error: {e}")
    
    return None

# Main app layout
st.title("Data Analysis Dashboard")

# File upload section
columns = upload_and_process_file()

# Sidebar for query input
with st.sidebar:
    st.header("Query Input")
    user_query = st.text_input("Enter your query", 
                              placeholder="e.g., Show me revenue by product",
                              key='user_query_input')
    
    # Query submission button
    if st.button("Run Query", key='run_query_button'):
        if columns:
            result_df = process_query(user_query, columns)
        else:
            st.error("Please upload a file first!")

# Display results and visualizations
if st.session_state.df is not None:
    st.write("Table Output:")
    st.table(st.session_state.df)
    
    # Generate insights
    insights = generate_summary_and_insights(st.session_state.df, user_query)
    st.write("Summary and Actionable Insights:")
    st.markdown(insights)
    
    # Create visualizations
    create_visualizations(st.session_state.df)

# Cleanup connection when the app is done
if st.session_state.conn is not None:
    st.session_state.conn.close()