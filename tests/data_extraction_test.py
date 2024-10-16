import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain.globals import set_llm_cache
from langchain_community.cache import InMemoryCache
from langchain_core.documents.base import Document

load_dotenv('../.env')
# print(os.environ.get("GOOGLE_API_KEY"))

from kgrag.data_extraction import Text2KG
# from kgrag.parse_pdf import PDFParserMarkdown, OCREngine
from kgrag.md_chunks import docs_to_md, chunk_md

from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings

set_llm_cache(InMemoryCache())
# set_llm_cache(RedisCache(ttl=3600))

# filepath = '/media/wali/D_Drive/DreamAI/KGRAG_POC/SampleDocs/Leadership-Etsko-Schuitema.pdf'
# filepath = "/media/wali/D_Drive/DreamAI/KGRAG_POC/SampleDocs/Technip Process Eng Guide.pdf"
# filepath = '/media/wali/D_Drive/Documents/FYP/FYP_Biomedical_FinalResearchPaper.pdf'
# filepath = '/home/wali/FYP_Biomedical_FinalResearchPaper.pdf'
# filepath = '/media/wali/D_Drive/Documents/FYP/Literature/Biomedical_relation_extraction_with_pre-trained_language_representations_and_minimal_task-specific_architecture.pdf'
# filepath = '/media/wali/D_Drive/Documents/FYP/Literature/Linking chemical and disease entities to ontologies by integrating PageRank with extracted relations from literature.pdf'
# filepath = '/media/wali/D_Drive/Documents/UniCoursesFiles/DeepLearning/Project/Literature/Deep Learning using CNNs for Ball-by-Ball Outcome Classification in Sports.pdf'
# filepath = '/media/wali/D_Drive/Documents/UniCoursesFiles/DeepLearning/Project/Literature/Sequence to Sequence - Video to Text.pdf'
filepath = "/media/wali/D_Drive/DreamAI/KGRAG_POC/SampleDocs/kosmos.pdf"

# emb_model=GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", task_type="semantic_similarity")
# emb_model.embed_query(filepath.split('/')[-1])
# "Research Paper: Biomedical Gene Disease Annotation Tool (BioGDAT)",

llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash-002", temperature=0)

# parser = PDFParserMarkdown(
#     pdf_path=filepath,
#     pages=2, #list(range(6, 20)),
#     ocr_engine=OCREngine.PYTESSERACT,
#     # llm=llm,
#     # ocr_llm=ChatGoogleGenerativeAI(model='models/gemini-1.5-flash', temperature=0.1)
# ) #, 

# doc_dicts: List[Dict[str, Any]] = parser.process_pdf_document()

# docs: List[Document] = [
#     Document(
#         page_content=doc['text'],
#         metadata={**doc['page_metadata'], **doc['doc_metadata']}
#     )
#     for doc in doc_dicts
# ]


mds = docs_to_md(filepath)
all_docs = []
for md in mds:
    chunks = chunk_md(text=md)

    docs: List[Document] = [
        Document(
            page_content=chunk['text'],
            metadata={
                "chunk_id": int(chunk_id), 
                "start": chunk['start'], 
                "end": chunk['end'], 
                "filepath": filepath,
                "source": f"{filepath.split('/')[-1]} Chunk {chunk_id}: {chunk['start']} - {chunk['end']}"
            }
        )
        for chunk_id, chunk in chunks.items()
    ]

    print(docs[-1])
    all_docs.extend(docs)

docs = all_docs

text2kg = Text2KG(
    llm=llm,
    emb_model=None, #emb_model,
    link_nodes=True,
    node_vector_similarity_threshold=0.90,
    subject="A finance agreement", #filepath.split('/')[-1].split('.')[0].replace('_',' '), #"Deep Learning and Artificial Intelligence",
    verbose=True
) 

text2kg.process_documents(docs)
