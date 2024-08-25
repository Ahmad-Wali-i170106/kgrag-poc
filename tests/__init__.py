from langchain_google_genai import ChatGoogleGenerativeAI
# Changes made in langchain_google_genai.function_utils._convert_pydantic_to_genai_function
# To fix using advanced schema with lists and nested schemas

from langchain_experimental.graph_transformers import LLMGraphTransformer

import os
from dotenv import load_dotenv

load_dotenv("../.env")
print(os.environ.get("GOOGLE_API_KEY"))

# llm = ChatOpenAI(model="gpt-4o-mini")
llm = ChatGoogleGenerativeAI(model='gemini-1.5-pro-latest')

llm_transformer = LLMGraphTransformer(llm=llm, strict_mode=False, node_properties=True)
