import os
from typing import List
from dotenv import load_dotenv
from langchain_core.documents.base import Document

load_dotenv('../.env')
# print(os.environ.get("GOOGLE_API_KEY"))

from kgrag.data_extraction import Text2KG
from kgrag.parsers_loaders import PDFLoader

from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings

# filepath = '/media/wali/D_Drive/DreamAI/KGRAG_POC/SampleDocs/Leadership-Etsko-Schuitema.pdf'
# filepath = "/media/wali/D_Drive/DreamAI/KGRAG_POC/SampleDocs/Technip Process Eng Guide_1_16.pdf"
# filepath = '/media/wali/D_Drive/Documents/FYP/FYP_Biomedical_FinalResearchPaper.pdf'
# filepath = '/home/wali/FYP_Biomedical_FinalResearchPaper.pdf'
filepath = '/media/wali/D_Drive/Documents/FYP/Literature/Biomedical_relation_extraction_with_pre-trained_language_representations_and_minimal_task-specific_architecture.pdf'
# filepath = '/media/wali/D_Drive/Documents/FYP/Literature/Linking chemical and disease entities to ontologies by integrating PageRank with extracted relations from literature.pdf'
# filepath = '/media/wali/D_Drive/Documents/UniCoursesFiles/DeepLearning/Project/Literature/Deep Learning using CNNs for Ball-by-Ball Outcome Classification in Sports.pdf'
# filepath = '/media/wali/D_Drive/Documents/UniCoursesFiles/DeepLearning/Project/Literature/Sequence to Sequence - Video to Text.pdf'

# emb_model=GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", task_type="semantic_similarity")
# emb_model.embed_query(filepath.split('/')[-1])
# "Research Paper: Biomedical Gene Disease Annotation Tool (BioGDAT)",

text2kg = Text2KG(
    llm=ChatGoogleGenerativeAI(model="models/gemini-1.0-pro", temperature=0),
    emb_model=None, #emb_model,
    disambiguate_nodes=True,
    node_vector_similarity_threshold=0.90,
    subject=filepath.split('/')[-1].split('.')[0].replace('_',' '), #"Deep Learning and Artificial Intelligence",
    verbose=True
) # 


loader = PDFLoader(filepath, extract_images=True)

docs: List[Document] = loader.load()

text2kg.process_documents(docs)
