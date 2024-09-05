# import langchain
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.output_parsers import JsonOutputParser
# from .llm import LLMClient

class PageMetadata(BaseModel):
    chapter_title: str
    chapter_num: int

def extract_chapter(text: str, model):
    parser = JsonOutputParser(pydantic_object=PageMetadata)
    template = "Please find if the input text from the user is the first page from a new chapter. If so, extract chapter number and chapter title from the input text. Otherwise, leave the answer empty.\n{format_instructions}\n{text}"
    # user_message = f"""{text}"""
    prompt = PromptTemplate.from_template(template=template, partial_variables={"format_instructions":parser.get_format_instructions()})

    chain = prompt | model | parser
    
    res = chain.invoke({"text": text})
    return res
