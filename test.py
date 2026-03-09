import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
import requests, pathlib
from langchain_community.utilities import SQLDatabase

load_dotenv()
HF_API_KEY = os.getenv("HF_TOKEN") or os.getenv("HF_API_KEY")

if not HF_API_KEY:
    raise RuntimeError(
        "HF token missing. Set HF_TOKEN (or HF_API_KEY) in your .env file."
    )

model = init_chat_model(
    "mistralai/Mistral-7B-Instruct-v0.2:featherless-ai",
    model_provider="huggingface",
    backend="endpoint",
    huggingfacehub_api_token=HF_API_KEY,
    temperature=0.7,
)

response = model.invoke("Reponds en une phrase: test de connexion distant OK.")
print(f"la réponse est : {response.content}")

url = "data/clean_dvf.db"
local_path = pathlib.Path("clean_dvf.db")

db = SQLDatabase.from_uri(f"sqlite:///{local_path}")

print(f"Dialect: {db.dialect}")
print(f"Available tables: {db.get_usable_table_names()}")
print(f'Sample output: {db.run("SELECT * FROM Artist LIMIT 5;")}')
