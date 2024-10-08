from typing import List, Union, Generator, Iterator
import os
from pydantic import BaseModel
from langchain_community.llms import Ollama
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain


class Pipeline:
    class Valves(BaseModel):
        DB_HOST: str
        DB_PORT: str
        DB_USER: str
        DB_PASSWORD: str
        DB_DATABASE: str
        DB_TABLE: str
        OLLAMA_HOST: str
        TEXT_TO_SQL_MODEL: str

    # Update valves/ environment variables based on your selected database
    def __init__(self):
        self.name = "Database RAG Pipeline"
        self.db = None

        # Initialize
        self.valves = self.Valves(
            **{
                "pipelines": ["*"],  # Connect to all pipelines
                "DB_HOST": os.getenv("DB_HOST", "localhost"),  # Database hostname (will not work if open-webui runs in docker as docker wont be able to access localhost port)
                "DB_PORT": os.getenv("DB_PORT", "5432"),  # Database port
                "DB_USER": os.getenv(
                    "DB_USER", "postgres"
                ),  # User to connect to the database with
                "DB_PASSWORD": os.getenv(
                    "DB_PASSWORD", "admin"
                ),  # Password to connect to the database with
                "DB_DATABASE": os.getenv(
                    "DB_DATABASE", "postgres"
                ),  # Database to select on the DB instance
                "DB_TABLE": os.getenv(
                    "DB_TABLE", "matches"
                ),  # Table(s) to run queries against
                "OLLAMA_HOST": os.getenv(
                    "OLLAMA_HOST", "http://localhost:11434"
                ),  # Make sure to update with the URL of your Ollama host, such as http://localhost:11434 or remote server address
                "TEXT_TO_SQL_MODEL": os.getenv(
                    "TEXT_TO_SQL_MODEL", "llama3.1:8b"
                ),  # Model to use for text-to-SQL generation
            }
        )

    def init_db_connection(self):
        # Update your DB connection string based on selected DB engine - current connection string is for Postgres
        self.db = SQLDatabase.from_uri(
            f"postgresql+psycopg2://{self.valves.DB_USER}:{self.valves.DB_PASSWORD}@{self.valves.DB_HOST}:{self.valves.DB_PORT}/{self.valves.DB_DATABASE}"
        )
        return self.db

    async def on_startup(self):
        # This function is called when the server is started.
        self.init_db_connection()

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        pass

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        
        llm = Ollama(
            model=self.valves.TEXT_TO_SQL_MODEL, base_url=self.valves.OLLAMA_HOST
        )
        db = self.db

        chain = create_sql_query_chain(llm, db)
        response = chain.invoke({"question": user_message})
        sql_query = response.split('SQLQuery: ')[1].strip()

        return db.run(sql_query)
