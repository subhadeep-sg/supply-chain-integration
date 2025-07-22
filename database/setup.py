from sqlalchemy import create_engine, text
import os
import pandas as pd
from dotenv import load_dotenv
import logging

load_dotenv()
database_url = os.getenv('DATABASE_URL')

if __name__ == '__main__':
    engine = create_engine(database_url)

    # Test connection
    with engine.connect() as conn:
        result = conn.execute(text("SELECT version();"))
        print(result.fetchone())

    query = "SELECT * FROM processed_data;"
    df = pd.read_sql_query(query, engine)

    print(df.to_markdown(index=False))