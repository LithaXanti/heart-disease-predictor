import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()

def get_engine():
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise ValueError("DATABASE_URL not found in .env")
    return create_engine(db_url, pool_pre_ping=True)

def load_data(query: str):
    engine = get_engine()
    return pd.read_sql(query, engine)