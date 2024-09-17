import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain.globals import set_llm_cache
from langchain_community.cache import InMemoryCache
from langchain_core.documents.base import Document

load_dotenv('../.env')
# print(os.environ.get("GOOGLE_API_KEY"))

from kgrag.data_extraction import Text2KG
from kgrag.parse_pdf import PDFParser, OCREngine

from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
# from langchain_community.document_loaders import PyMuPDFLoader

set_llm_cache(InMemoryCache())
# set_llm_cache(RedisCache(ttl=3600))

filepath = '/media/wali/D_Drive/DreamAI/KGRAG_POC/SampleDocs/Leadership-Etsko-Schuitema.pdf'
# filepath = "/media/wali/D_Drive/DreamAI/KGRAG_POC/SampleDocs/Technip Process Eng Guide.pdf"
# filepath = '/media/wali/D_Drive/Documents/FYP/FYP_Biomedical_FinalResearchPaper.pdf'
# filepath = '/home/wali/FYP_Biomedical_FinalResearchPaper.pdf'
# filepath = '/media/wali/D_Drive/Documents/FYP/Literature/Biomedical_relation_extraction_with_pre-trained_language_representations_and_minimal_task-specific_architecture.pdf'
# filepath = '/media/wali/D_Drive/Documents/FYP/Literature/Linking chemical and disease entities to ontologies by integrating PageRank with extracted relations from literature.pdf'
# filepath = '/media/wali/D_Drive/Documents/UniCoursesFiles/DeepLearning/Project/Literature/Deep Learning using CNNs for Ball-by-Ball Outcome Classification in Sports.pdf'
# filepath = '/media/wali/D_Drive/Documents/UniCoursesFiles/DeepLearning/Project/Literature/Sequence to Sequence - Video to Text.pdf'

# emb_model=GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", task_type="semantic_similarity")
# emb_model.embed_query(filepath.split('/')[-1])
# "Research Paper: Biomedical Gene Disease Annotation Tool (BioGDAT)",

llm = ChatGoogleGenerativeAI(model="models/gemini-1.0-pro", temperature=0)

parser = PDFParser(
    filepath,
    OCREngine.PYTESSERACT,
    llm,
    ChatGoogleGenerativeAI(model='models/gemini-1.5-flash', temperature=0.1)
) #, 

# docs: List[Document] = loader.load()
doc_dicts: List[Dict[str, Any]] = parser.process_pdf_document(num_of_pages=21)

docs: List[Document] = [
    Document(
        page_content=doc['text'],
        metadata={**doc['page_metadata'], **doc['doc_metadata']}
    )
    for doc in doc_dicts
]
print(docs[-1])


text2kg = Text2KG(
    llm=llm,
    emb_model=None, #emb_model,
    disambiguate_nodes=False,
    node_vector_similarity_threshold=0.90,
    subject=filepath.split('/')[-1].split('.')[0].replace('_',' '), #"Deep Learning and Artificial Intelligence",
    verbose=True
) 

text2kg.process_documents(docs)
