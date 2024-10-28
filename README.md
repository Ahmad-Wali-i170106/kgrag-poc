# KGRAG POC

This repository contains code to create a knowledge graph in [Neo4j](https://neo4j.com) from text/PDF documents, webpages etc.
This knowledge graph can then be queried using search queries by the user.

## Setting Up The Environment

### Neo4j Installation

You will need access to a Neo4j Server to use this. There are 3 possible [deployments options](https://neo4j.com/docs/deployment-options/):

1. [Neo4j AuraDB](https://neo4j.com/product/auradb/) - Neo4j's Fully-managed Database service

2. [Neo4j Graph Database Self-Managed](https://neo4j.com/product/neo4j-graph-database/) - There are both community and enterprise versions available. You can find detailed installation instructions [here](https://neo4j.com/docs/operations-manual/current/installation/)

3. [Neo4j Desktop](https://neo4j.com/docs/desktop-manual/) - Not suitable for production environments

You can find and install Neo4j [here](https://neo4j.com/deployment-center/)

Following are the python packages and their versions you will need to run this code:

### Required Packages

- loguru = "^0.7.2"
- python-dotenv = "^1.0.1"
- pandas = "^2.2.2"
- neo4j = "^5.23.1"
- sentence-transformers = "^3.0.1"
- pymupdf = "^1.24.9"
- pymupdf4llm = "^0.0.16"
- langchain = "^0.2.11"
- langchain-community = "^0.2.10"
- langchain-experimental = "^0.0.63"

### Optional Packages

You only need either of `langchain-openai` or `langchain-google-genai` if you need to use an LLM or an embedding model from either of these providers
Similarly, you only need `rapidocr-onnxruntime` or `pytesseract` if you choose to use them to OCR the documents.

- rapidocr-onnxruntime = "^1.3.24"
- langchain-openai = "^0.1.20"
- langchain-google-genai = "^1.0.8"
- pytesseract = "^0.3.10"

Version "^1.3.4" means ">=1.3.4" and "<2.0.0". This applies to all the versions mentioned above. 

### Environment Variables

You may need to set the following environment variables:

1. NEO4J_URL (Default="bolt://localhost:7687"), 
2. NEO4J_USERNAME (Default="neo4j"), 
3. NEO4J_PASSWORD (Default=None), 
4. NEO4J_DATABASE (Default="neo4j")
(The above are for establishing a Neo4j Connection and can be passed as arguments as well)
5. MODELS_CACHE_FOLDER (Default=None):- 
    I'm using SentenceTransformer model "all-MiniLM-v6-L2" for creating embeddings to calculate vector similarity.
    This is just the path to the folder to use as the cache folder to store the model
6. GOOGLE_API_KEY (For ChatGoogleGenerativeAI) or OPENAI_API_KEY (for ChatOpenAI) or any other API key for env variable if you want to use some other LLM

## Creating the Knowledge Graph

### Example Code

You can create a knowledge graph from a list of documents (`langchain_core.documents.base.Document`) as follows:

```python

from kgrag.data_extraction import Text2KG
from kgrag.md_chunks import docs_to_md, chunk_md
from langchain_core.documents.base import Document
from langchain_google_genai import ChatGoogleGenerativeAI

# Replace with llm you want to use
llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash-002", temperature=0)

# Can either accept neo4j_url, neo4j_username, neo4j_password & neo4j_database arguments
# Or set the NEO4J_URL, NEO4J_USERNAME, NEO4J_PASSWORD and NEO4J_DATABASE environment variables
# if the above mentioned environment variables have not been set
text2kg = Text2KG(
    llm=llm,
    emb_model=None, # By Default, SentenceTransformer's all-MiniLM-v6-L2 is used but you can provide any other embedding model
    link_nodes=True,
    node_vector_similarity_threshold=0.90,
    subject=filepath.split('/')[-1].split('.')[0].replace('_',' ').replace('-',' '), # Subject can be anything or nothing - filename works well for most cases
    verbose=True
)

'''
Parse or read the PDF file
Can use any parser as long as some conditions are observed in the output or process_documents input:
    1. Must be in langchain_core.documents.base.Document format
    2. Must contain the following keys in doc.metadata:
        1. page
        2. filename/filepath or source
        3. chunk_id
'''
if filepath.endswith(".pdf"):
    md = docs_to_md(filepath)[0]
elif filepath.endswith('.md') or filepath.endswith('.txt'):
    md = ''
    with open(filepath, "r") as f:
        md = f.read()

chunks = chunk_md(text=md, separators=[r"\* \* \*"])

docs: List[Document] = [
    Document(
        page_content=chunk['text'],
        metadata={
            "chunk_id": int(chunk_id), 
            "start": chunk['start'], 
            "end": chunk['end'], 
            "filepath": filepath,
            "source": f"{filepath.split('/')[-1]} Page {chunk_id}"
        }
    )
    for chunk_id, chunk in chunks.items()
]


text2kg.process_documents(docs, use_existing_node_types=False)
```

### Functions Parameters

#### Text2KG Constructor

- llm (language_core.language_models.BaseLanguageModel): The LLM to use

- emb_model (language_core.language_models.Embeddings | sentence_transformers.SentenceTransformer | None): The embeddings model. Default=None.

- data_ext_prompt (str): The prompt to use in the data extraction chain. Default=DATA_EXTRACTION_SYSTEM.

- link_nodes (bool): Whether to link the nodes or not. Default=True

- subject (str | None): Optional. The subject or topic that the documents provided will likely belong to. Default=None

- neo4j_url (str): The full URL to the Neo4j server. Default="bolt://localhost:7687"

- neo4j_username (str): The username of the Neo4j authenticated user. Default="neo4j"

- neo4j_password (str): The password of the Neo4j server authentication. Default=None

- neo4j_database (str): The name of the Neo4j database to use. Default="neo4j".

- embed_model_name (str): Only used when emb_model is None. The name of the SentenceTransformer model to use for generating embeddings.Default="sentence transformers/all-MiniLM-L6-v2"

- node_vector_similarity_threshold (float): The minimum vector similarity score to match a new node to an existing node. Default=0.85 (0<=score<=1.0)

- node_id_fulltext_similarity_threshold (float): The minimum fulltext similarity score to match the node ID of a node mentioned in a relationship or document to an existing node. Default=1.0
  
- verbose (bool): Whether to log/print the progress of the KG creation process. Default=False

#### Text2KG.process_documents

- docs (list[langchain_core.documents.base.Document]): The list of documents to process

- use_existing_node_types (bool): Whether to use the node types already present in the KG in the data extraction prompt. Slightly limits the LLM in the node types of the nodes that it can extract.

## Querying the KG

### Example Code

The following code shows how to query the KG using the `KGSearch.retrieve` method:

```python
from langchain_google_genai import ChatGoogleGenerativeAI

from kgrag.kg_search import KGSearch

# You can use any other LLM here as long as it is of type `langchain_core.language_models.BaseLanguageModel`
llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash-002", temperature=0)

kg_search = KGSearch(
    ent_llm=llm2,
    cypher_llm=llm,
    cypher_examples_json="examples.json", # The JSON file containing the few-shot examples to use in the prompt
    fulltext_search_top_k=5,
    vector_search_top_k=8,
    vector_search_min_score=0.8
)

query = input("Please enter your search query: ")

docs_string = kg_search.retrieve_as_string(
    query, 
    # nresults=400,
    use_fulltext_search=True, 
    use_vector_search=True,
    generate_cypher=False
)
'''
OR get 3 lists separately using kg_search.retrieve - retrieve_as_string is only a wrapper for this function that combines all these results into a single string result

rels, docs, gen_cypher_results = kg_search.retrieve(
    query, 
    nresults=30,
    use_fulltext_search=True, 
    use_vector_search=True,
    generate_cypher=False,
    return_chunk_ids=True
)
rels: All the triples that the entities/nodes in the query are involved in 
     Empty list is returned if `use_fulltext_search=False` && `use_vector_search=False`
docs: All the documents that contain any entities mentioned in the query
     Empty list is returned if `use_fulltext_search=False` && `use_vector_search=False`
     If return_chunk_ids=True, returns chunk_ids as set during text2kg conversion / data insertion process
gen_cypher_results: List of string/JSON results from the generated cypher
     Empty list is returned if `generate_cypher=False`

If no entities are found using vector or fulltext search, then a cypher is generated regardless of the value of `generate_cypher`
And results are returned in the gen_cypher_results
WARNING: Using `generate_cypher` is unreliable and error-prone. Needs more examples in 'examples.json' to improve reliability
'''

# Following is one possible way to use these results:
q = f"""Answer the given query as best as you can. Use the given search results to assist you with answering the query
Query: {query}

Context:
{docs_string}

Answer:"""
ans = llm.invoke(q)
print(ans.content)
```

### Functions Parameters

#### KGSearch Constructor
   
- ent_llm (langchain_core.language_models.BaseLanguageModel): The Language Model to use to extract entities

- cypher_llm (langchain_core.language_models.BaseLanguageModel): The language model to use to generate cypher

- sim_model (sentence_transformers.SentenceTransformer | langchain_core.embeddings.Embeddings): 
            The embedding model to use to generate the embeddings for calculating vector similarity. 
            Should be same as the one used  in the KG creation process. Default=None
- cypher_examples_json (str): The path to the JSON file containing a list of cypher examples in the format {"question": <query>, "cypher": <cypher>}

- neo4j_url (str): The full url to the Neo4j server. If NEO4J_URL environment variable is set, uses that instead. Default="bolt://localhost:7687"

- neo4j_username (str): The username to provide for authentication to the Neo4j server. If NEO4J_USERNAME env variable is set, uses that instead. Default="neo4j"

- neo4j_password (str): The password to provide for authentication to the Neo4j server. If NEO4J_PASSWORD env variable is set, uses that instead. Default=None

- neo4j_database (str): The neo4j database name to query. If NEO4J_DATABASE env variable is set, uses that instead. Default="neo4j"

- fulltext_search_max_difference (float): The maximum score difference allowed between 2 consecutive fulltext search results. Default=2.0. Use to make sure that search results are always close to each other.

- fulltext_search_min_score (float): The minimum score of a fulltext search result for it to be considered. Default=1.0

- fulltext_search_top_k (int): The top 'k' fulltext search results to consider. Default=10

- vector_search_top_k (int): The top 'k' vector search results to consider. Default=15

- vector_search_min_score (float): The minimum score of vector search result for it to be considered. Default=0.75 where 0 <= score <= 1.0.

- max_cypher_fewshot_examples (int): The maximum number of similar cypher fewshot examples given in `cypher_examples_json` to choose dynamically. Default=15

- embed_model_name (str): Only used when sim_model is None. The name of the SentenceTransformer model to use for generating embeddings. Default="sentence-transformers/all-MiniLM-L6-v2"

#### KGSearch.retrieve

- query (str): The search query

- nresults (int): The number of results to return. Default=None (Return all results)

- use_fulltext_search (bool): Whether to use fulltext search to search for nodes. Default=True

- use_vector_search (bool): Whether to use vector search to search for nodes. Default=True

- generate_cypher (bool): Whether to generate cypher from user query to get results. Default=False

- return_chunk_ids (bool): Whether to return the chunk IDs of the chunks/docs that are found. Otherwise, return the text content of the docs/chunks
