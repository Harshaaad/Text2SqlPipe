"""
title: LangChain Text to SQL Pipeline
author: Harshad
date: 2024-08-11
version: 1.1
license: MIT
description: A pipeline for using text-to-SQL for retrieving relevant information from a database using the LangChain library.
requirements: langchain, langchain-community, psycopg2-binary, langchain-core
"""

from typing import List, Union, Generator, Iterator
import os
from pydantic import BaseModel
from langchain_community.llms import Ollama
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


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
        self.name = "Text to SQL Pipeline"
        self.db = None

        # Initialize
        self.valves = self.Valves(
            **{
                "pipelines": ["*"],  # Connect to all pipelines
                "DB_HOST": os.getenv("DB_HOST", "host.docker.internal"),  # Database hostname (will not work if open-webui runs in docker as docker wont be able to access localhost port)
                "DB_PORT": os.getenv("DB_PORT", 5432),  # Database port
                "DB_USER": os.getenv(
                    "DB_USER", "postgres"
                ),  # User to connect to the database with
                "DB_PASSWORD": os.getenv(
                    "DB_PASSWORD", "mysecetpassword"
                ),  # Password to connect to the database with
                "DB_DATABASE": os.getenv(
                    "DB_DATABASE", "postgres"
                ),  # Database to select on the DB instance
                "DB_TABLE": os.getenv(
                    "DB_TABLE", "employees"
                ),  # Table(s) to run queries against
                "OLLAMA_HOST": os.getenv(
                    "OLLAMA_HOST", "http://host.docker.internal:11434"
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

        print(db.get_usable_table_names())

        template = '''Given an input question, first create a syntactically correct {top_k} query to run, then look at the results of the query and return the answer.
                    Use the following format, only generate the sql query. Do not add any statements or characters before or after that:

                    "SQL Query to run"


                    Only use the following tables:

                    {table_info}.

                    Question: {input}'''
        
        prompt = PromptTemplate.from_template(template)

        chain = create_sql_query_chain(llm, db, prompt)
        response = chain.invoke({"question": user_message})
        result = db.run(response)

        _template = """You are a helpful chat assistant. Given an SQL Query {response} and the result obtained after
               running it agains a database {result}, generate user friendly, readable reply, being aware of
               the context. Be concise, try to conclude in a single sentence, use maximum 3 sentences if
               the result is complex"""

        _prompt = PromptTemplate.from_template(_template)

        _chain = _prompt | llm | StrOutputParser()

        _response = _chain.invoke({"response":response, "result":result})

        return _response
