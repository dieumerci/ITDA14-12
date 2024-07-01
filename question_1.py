import sqlite3
import pandas as pd

def create_sqlite_connection(csv_file_path, db_file_path, table_name):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(csv_file_path)

    # Create a connection to the SQLite database
    conn = sqlite3.connect(db_file_path)
    cursor = conn.cursor()

    # Upload the DataFrame to the database as a table
    df.to_sql(table_name, conn, if_exists='replace', index=False)

    # Verify by fetching the first few rows
    cursor.execute(f"SELECT * FROM {table_name} LIMIT 5")
    rows = cursor.fetchall()

    # Close the connection
    conn.close()

    return rows

# Define file paths and table name
csv_file_path = 'heart.csv'
db_file_path = 'heart.db'
table_name = 'heart_data'

# Create and set up the database
sample_data = create_sqlite_connection(csv_file_path, db_file_path, table_name)

# Print sample data to verify
for row in sample_data:
    print(row)

# Function to connect to the SQLite database and perform queries
def connect_and_query(db_file_path, query):
    # Create a connection to the SQLite database
    conn = sqlite3.connect(db_file_path)
    cursor = conn.cursor()

    # Execute the query
    cursor.execute(query)
    results = cursor.fetchall()

    # Close the connection
    conn.close()

    return results

# Example query to fetch data from the table
query = f"SELECT * FROM {table_name} LIMIT 10"
query_results = connect_and_query(db_file_path, query)

# Print query results
for row in query_results:
    print(row)
    