from dotenv import load_dotenv
from langchain.globals import set_llm_cache
from langchain_community.cache import InMemoryCache
load_dotenv('../.env')
from kgrag.parse_pdf import PDFParser, OCREngine

from langchain_google_genai import ChatGoogleGenerativeAI


set_llm_cache(InMemoryCache())

# filepath = '/media/wali/D_Drive/DreamAI/KGRAG_POC/SampleDocs/Leadership-Etsko-Schuitema.pdf'
filepath = "/media/wali/D_Drive/DreamAI/KGRAG_POC/SampleDocs/Technip Process Eng Guide.pdf"
# filepath = '/media/wali/D_Drive/Documents/FYP/FYP_Biomedical_FinalResearchPaper.pdf'
# filepath = '/home/wali/FYP_Biomedical_FinalResearchPaper.pdf'
# filepath = '/media/wali/D_Drive/Documents/FYP/Literature/Biomedical_relation_extraction_with_pre-trained_language_representations_and_minimal_task-specific_architecture.pdf'
# filepath = '/media/wali/D_Drive/Documents/FYP/Literature/Linking chemical and disease entities to ontologies by integrating PageRank with extracted relations from literature.pdf'
# filepath = '/media/wali/D_Drive/Documents/UniCoursesFiles/DeepLearning/Project/Literature/Deep Learning using CNNs for Ball-by-Ball Outcome Classification in Sports.pdf'
# filepath = '/media/wali/D_Drive/Documents/UniCoursesFiles/DeepLearning/Project/Literature/Sequence to Sequence - Video to Text.pdf'

parser = PDFParser(
    filepath,
    OCREngine.PYTESSERACT,
    ChatGoogleGenerativeAI(model="models/gemini-1.0-pro", temperature=0),
    ChatGoogleGenerativeAI(model='models/gemini-1.5-flash', temperature=0.1)
) #, 

docs = parser.process_pdf_document(num_of_pages=5)
for i, doc in enumerate(docs):
    print(f"---------------Page {i+1}---------------")
    print(doc['text'])
    print()