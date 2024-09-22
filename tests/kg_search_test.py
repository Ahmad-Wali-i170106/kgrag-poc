import os
from dotenv import load_dotenv
from langchain.globals import set_llm_cache
from langchain_community.cache import InMemoryCache

load_dotenv("../.env")

from langchain_google_genai import ChatGoogleGenerativeAI

from kgrag.kg_search import KGSearch

set_llm_cache(InMemoryCache())

llm = ChatGoogleGenerativeAI(model="models/gemini-1.0-pro", temperature=0)

print(os.getcwd())

kg_search = KGSearch(
    ent_llm=llm,
    cypher_llm=llm,
    cypher_examples_json="/media/wali/D_Drive/DreamAI/KGRAG_POC/kgrag/examples.json"
)

results = kg_search.retrieve("How does relation extraction work?")
print(results)