import os

import pymysql
from dotenv import load_dotenv


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.dirname(CURRENT_DIR)
PROJECT_ROOT = os.path.dirname(APP_DIR)

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))


def get_db():
    return pymysql.connect(
        host=os.getenv("DB_HOST"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME"),
        charset=os.getenv("DB_CHARSET", "utf8mb4"),
    )
