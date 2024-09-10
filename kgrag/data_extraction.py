import os
from typing import List, Dict, Any, Optional, Tuple, Generator

import neo4j
from numpy import double
from sentence_transformers import SentenceTransformer

from langchain.pydantic_v1 import BaseModel

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseLanguageModel
from langchain_core.embeddings import Embeddings

from langchain_core.runnables import RunnableSerializable

from langchain_community.graphs.neo4j_graph import value_sanitize

from kgrag.data_schema_utils import * 
from kgrag.data_disambiguation import DataDisambiguation

def chunks(lst: List, n: int) -> Generator[list, Any, None]:
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

# - **Suggested Relationship Types**: Following are some suggested relationship types/labels that you can use: {rel_types}
SYSTEM_PROMPT = """# Knowledge Graph Instructions
## 1. Overview
You are a top-tier algorithm designed for extracting information in structured formats to build a knowledge graph.
You must extract the information from consecutive samples of text from a much larger text, most likely a book or short tutorial.
You must extract information that seems most relevant to the overall subject theme of the text.
{subject}
- **Nodes** represent entities and concepts. They're akin to Wikipedia nodes. They are assigned a type which is akin to entity type for entities. In general, proper nouns should be used for creating nodes.
- **Relationships** represent the relations or edges between these **nodes**.
- The aim is to achieve simplicity and clarity in the knowledge graph, making it accessible for a vast audience.
## 2. Labeling Nodes
- **Node IDs**: Never utilize integers as node IDs. Node IDs should be names or human-readable identifiers found in the text. Node IDs should be clear and unambiguous. 
- **Expand Acronyms**: Do NOT use abbreviations, acronyms or pronouns for Node IDs e.g. use 'Artificial Intelligence' instead of 'AI' or 'This Technology'. Expand any acronyms based on the context and subject of the text.
- **Node IDs Naming Convention**: Node IDs should always be written in Title Case.
- **Node Type Naming Convention**: Node types should always be written in PascalCase.
- **Entity/Node Type**: Choose a descriptive node/entity type/label that best describes the context in which that entity appears.
- - You are encouraged to be descriptive with the entity types.
- - Following are some existing node/entity types that were extracted from previous samples of the same document:\n{node_types}
## 3. Labeling Relationships
- **Relationship Type Naming Convention**: Relationship types should be written in SCREAMING_SNAKE_CASE.
## 5. Coreference Resolution
- **Maintain Entity Consistency**: When extracting entities, it's vital to ensure consistency.
- If an entity, such as "John Doe", is mentioned multiple times in the text but is referred to by different names or pronouns (e.g., "Joe", "he"), 
always use the most complete identifier for that entity throughout the knowledge graph. In this example, use "John Doe" as the entity ID.  
- If an entity, such as 'Artificial Intelligence' is mentioned multiple times in the text but is referred to by different names, pronouns or acronyms (e.g., "AI", "A.I"),
always use the most complete identifier for that entity throughout the knowledge graph. In this example, use "Artificial Intelligence" as the entity ID.
Remember, the knowledge graph should be coherent and easily understandable, so maintaining consistency in entity references is crucial. 
## 6. Strict Compliance
Adhere to the rules strictly. Non-compliance will result in termination."""

# ## 4. Handling Numerical Data and Dates
# - Numerical data, like age or other related information, should be incorporated as attributes or properties of the respective nodes.
# - **No Separate Nodes for Dates/Numbers**: Do not create separate nodes for dates or numerical values. Always attach them as attributes or properties of nodes.
# - **Property Format**: Properties must be in a key-value format.
# - **Quotation Marks**: Never use escaped single or double quotes within property values.
# - **Naming Convention**: Use camelCase for property keys.
# - **Type Property**: NEVER include the `type` property in the list of properties.



def get_extraction_chain(llm: BaseLanguageModel) -> RunnableSerializable[Dict, Dict | BaseModel]:

    from langchain_google_genai.llms import _BaseGoogleGenerativeAI

    if isinstance(llm, _BaseGoogleGenerativeAI) and 'gemini-1.0' in llm.model:
        prompt: ChatPromptTemplate = ChatPromptTemplate.from_messages(
        [
            ("human", SYSTEM_PROMPT),
            ("human", "Use the given format to extract information from the following input which is a small sample from a much larger text belonging to the same subject matter: {input}"),
            ("human", "Tip: Make sure to answer in the correct format"),   
        ])
    else:
        prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "Use the given format to extract information from the following input which is a small sample from a much larger text belonging to the same subject matter: {input}"),
            ("human", "Tip: Make sure to answer in the correct format"),
        ])
    
    return prompt | llm.with_structured_output(KnowledgeGraph)


class Text2KG:

    def __init__(
            self, 
            llm: BaseLanguageModel, 
            emb_model: Embeddings | None = None, 
            disambiguate_nodes: bool = False,
            **kwargs
        ) -> None:

        
        url: str = os.environ.get("NEO4J_URL", kwargs.get("neo4j_url", "bolt://localhost:7687"))
        username: str = os.environ.get("NEO4J_USERNAME", kwargs.get("neo4j_username", "neo4j"))
        password: str = os.environ.get("NEO4J_PASSWORD", kwargs.get("neo4j_password", None))
        if password is None:
            auth = None
        else:
            auth:Tuple[str, str] = (username, password)
        
        self._driver = neo4j.GraphDatabase.driver(url,auth=auth)
        self._database: str = os.environ.get("NEO4J_DATABASE", kwargs.get("neo4j_database", "neo4j"))
        try:
            self._driver.verify_connectivity()
        except neo4j.exceptions.ServiceUnavailable:
            raise ValueError(
                "Could not connect to Neo4j database. "
                "Please ensure that the url is correct"
            )
        except neo4j.exceptions.AuthError:
            raise ValueError(
                "Could not connect to Neo4j database. "
                "Please ensure that the username and password are correct"
            )

        self.verbose: bool = kwargs.get("verbose", False)
        self.timeout: int | None = kwargs.get("timeout", None)
        
        # Should be approximately equal to the minimum number of nodes present in a single doc
        self._node_batch_size: int = kwargs.get("node_batch_size", 1)
        self._sim_thresh: float = kwargs.get("node_vector_similarity_threshold", 0.85)
        self._ft_sim_thresh: float = kwargs.get("node_id_fulltext_similarity_threshold", 1.0)
        text_subject: str | None = kwargs.get("subject", None)
        if text_subject is not None and len(text_subject.strip()) > 0:
            self.text_subject: str = f"""The input text has the following subject/topic/filename: `{text_subject}`.
You must extract information that appears most relevant to this provided subject."""

        self.data_ext_chain: RunnableSerializable[Dict, Dict | BaseModel] = get_extraction_chain(llm)
        if emb_model is None:
            emb_model = SentenceTransformer(
                kwargs.get("embed_model_name", 'sentence-transformers/all-MiniLM-L6-v2'), 
                cache_folder=os.environ.get("MODELS_CACHE_FOLDER", None),
                tokenizer_kwargs={"clean_up_tokenization_spaces": False}
            )

        self.emb_model: SentenceTransformer | Embeddings = emb_model

        # self._disambiguate: bool = disambiguate_nodes
        if disambiguate_nodes:
            self.disambiguator = DataDisambiguation(llm=kwargs.get("disam_llm", None) or llm)
        else:
            self.disambiguator = None
        
        self.__create_indexes()

    def __create_indexes(self) -> None:
        ftquery = """CREATE FULLTEXT INDEX IDsAndAliases IF NOT EXISTS FOR (n:Node) ON EACH [n.id, n.alias]
OPTIONS { indexConfig: {
    `fulltext.analyzer`: 'standard',
    `fulltext.eventually_consistent`:false 
} };"""
        
        vquery = """CREATE VECTOR INDEX IDsVectors IF NOT EXISTS FOR (n:Node) ON n.embedding
OPTIONS { indexConfig: {
    `vector.similarity_function`: 'cosine'
} };
"""

        constraints: list[str] = [
            # "CREATE CONSTRAINT unique_id IF NOT EXISTS FOR (n:Node) REQUIRE n.id IS UNIQUE;",
            "CREATE CONSTRAINT unique_source IF NOT EXISTS FOR (d:DocumentPage) REQUIRE d.source IS UNIQUE;"
        ]

        self.graph_query(ftquery) # Create the fulltext index
        self.graph_query(vquery) # Create the vector index
        for cq in constraints:
            self.graph_query(cq)

    def graph_query(self, query: str, params: dict = {}) -> List[Dict[str, Any]]:
        """Query Neo4j database.

        Args:
            query (str): The Cypher query to execute.
            params (dict): The parameters to pass to the query.

        Returns:
            List[Dict[str, Any]]: The list of dictionaries containing the query results.
        """
        from neo4j import Query
        from neo4j.exceptions import CypherSyntaxError

        with self._driver.session(database=self._database) as session:
            try:
                data: neo4j.Result = session.run(Query(text=query, timeout=self.timeout), params)
                json_data: List[Dict[str, Any]] = [r.data() for r in data]
                json_data = [value_sanitize(el) for el in json_data]
                return json_data
            except CypherSyntaxError as e:
                raise ValueError(f"Generated Cypher Statement is not valid\n{e}")

    def _embed(self, sents: List[str]) -> List[List[float | double]]:
        if isinstance(self.emb_model, SentenceTransformer):
            return [list(sent) for sent in self.emb_model.encode(sents)]
        elif isinstance(self.emb_model, Embeddings):
            return self.emb_model.embed_documents(sents)

    def _upsert_nodes(self, nodes: List[Node]) -> None:
        '''
        Create new nodes in the Neo4j KG from the given list of nodes

        Performs an additional step of resolving and merging each new node to an existing node 
        if it exceeds a certain similarity threshold with an existing node.
        The ID of the new node is added to an alias property of the existing node.
        If no match with an existing node is found, a new node is created.

        In affect, it is a more complicated form of MERGE clause that uses semantic similarity
        to only create nodes that are considered semantically different.
        '''

        #  AND apoc.label.exists(n, node.type)
        query = """UNWIND $nodes AS nd
CALL (nd) {
    CALL db.index.vector.queryNodes('IDsVectors', 1, nd.embedding) YIELD node AS n, score
    WHERE score > $similarity_threshold
    RETURN collect(n)[0] AS n
}
CALL apoc.do.when(n IS NULL,
    'CALL apoc.merge.node(["Node", $ntype], {id: $nid}) YIELD node RETURN node',
    'CALL apoc.create.addLabels($n, [$ntype]) YIELD node RETURN node',
    {n: n, ntype: nd.type, nid: nd.id}
)
YIELD value AS nn
WITH nd AS nd, nn.node AS nn
CALL db.create.setNodeVectorProperty(nn, 'embedding', nd.embedding)
SET nn += nd.properties
FOREACH(x in CASE WHEN nn.alias IS NULL OR NOT (nd.id IN nn.alias OR nd.id = nn.id) THEN [1] END | 
    SET nn.alias = COALESCE(nn.alias,[]) + nd.id )
RETURN DISTINCT nn;
"""

        # Use Node ID, Node Type & string properties to create a string to generate embeddings from
        # That will be used to calculate similarity
        emb_strings: List[str] = []
        for node in nodes:
            # props: str = ', '.join([f"{p.key}: {p.value}" for p in node.properties if isinstance(p.value, str) or isinstance(p.value, int)])
            # if len(props) > 0:
            #     emb_strings.append(f"{node.id} {{{props}}}")
            # else:
            emb_strings.append(node.id)
        embeddings: List[List[float | double[Any]]] = self._embed(emb_strings)
        nodes = [
            {
                "id": node.id,
                "type": node.type,
                "properties": {}, #{p.key: p.value for p in node.properties},
                "embedding": list(embeddings[i])

            }
            for i, node in enumerate(nodes)
        ]
        for bnodes in chunks(nodes, self._node_batch_size):
            
            result: List[Dict[str, Any]] = self.graph_query(query, params={"nodes": bnodes, "similarity_threshold": self._sim_thresh})
            if self.verbose:
                print(result)


    def _upsert_rels(self, rels: List[Relationship]) -> None:
        '''
        Creates new relationships between existing nodes in the KG
        '''
        query = """UNWIND $rels AS r
CALL (r) {
    CALL db.index.fulltext.queryNodes('IDsAndAliases', r.start_node_id) YIELD node, score
    WHERE score > $similarity_threshold
    RETURN node AS start_node LIMIT 1
}
CALL (r) {
    CALL db.index.fulltext.queryNodes('IDsAndAliases', r.end_node_id) YIELD node, score
    WHERE score > $similarity_threshold
    RETURN node AS end_node LIMIT 1
}
WITH r, start_node, end_node
WHERE start_node <> end_node
CALL apoc.merge.relationship(start_node, r.type, r.properties, {}, end_node, {}) YIELD rel
RETURN DISTINCT start_node{.id,.alias}, rel, end_node{.id, .alias};"""

        rels = [
            {
                "start_node_id": ' '.join([r.strip() for r in rel.start_node_id.split(' ') if len(r.strip()) > 0]),
                "end_node_id": ' '.join([r.strip() for r in rel.end_node_id.split(' ') if len(r.strip()) > 0]),
                "type": rel.type,
                "properties": {} #{p.key: p.value for p in rel.properties}
            }
            for rel in rels
        ]
        new_rels: List[Dict[str, Any]] = self.graph_query(query, params={"rels": rels, "similarity_threshold": self._ft_sim_thresh})
        if self.verbose:
            print(new_rels)

    def _create_text_nodes(self, docs: List[Document]) -> None:
        '''
        Creates new Text nodes also called DocumentPage nodes
        '''

        query = """UNWIND $mentions AS mention
MERGE (d: DocumentPage {source: $source})
SET
    d = $properties,
    d.source = $source,
    d.text = $text
WITH mention, d
CALL (mention) {
    CALL db.index.fulltext.queryNodes('IDsAndAliases', mention.id) YIELD node, score
    WHERE score > $similarity_threshold AND apoc.label.exists(node, mention.type)
    RETURN node
}

MERGE (d)-[:MENTIONS]-(node)
RETURN DISTINCT d.source, node.alias"""

        for doc in docs:
            # Prepare the entity mentions in this particular doc
            mentions: List[str] = doc.metadata.pop("mentions", [])
            mentions = [
                {
                    "id": ' '.join([f"{mid}" for mid in m.split('::')[0].strip().split(' ')]), 
                    "type": m.split('::')[-1].strip() if len(m.split('::')) > 1 else "Node"
                }
                for m in mentions
            ]

            source: str = doc.metadata.pop("source", '')
            if source is not None or len(source) > 0:
                if not "page" in source:
                    source = f"{source.split('/')[-1]} Page: {str(doc.metadata.get('page', 0))}"
            elif 'filename' in doc.metadata:
                source = f"{doc.metadata.get('filename', '')} Page: {str(doc.metadata.get('page', 0))}"
            elif 'filepath' in doc.metadata:
                source = f"{doc.metadata.get('filepath', '')} Page: {str(doc.metadata.get('page', 0))}"
            else:
                for k, v in doc.metadata.items():
                    if 'file' in k or 'source' in k:
                        source = f"{v} Page: {str(doc.metadata.get('page', 0))}"
                        break
            # params = {
            #     "text": doc.page_content,
            #     "source": doc.metadata.get('source'),
            #     "properties": doc.metadata
            # }
            # new_doc_node = self.graph_query(cquery, params=params)
            # if self.verbose:
            #     print(new_doc_node)
            params = {
                "text": doc.page_content,
                "source": source,
                "mentions": mentions, 
                "properties": doc.metadata,
                "similarity_threshold": self._ft_sim_thresh
            }
            result: List[Dict[str, Any]] = self.graph_query(query, params=params)
            if self.verbose:
                print(result)

    
    def __get_existing_labels(self) -> Tuple[set, set]:

        labels_query = """CALL apoc.meta.data() YIELD label, elementType
WHERE NOT label IN $EXCLUDED_LABELS
RETURN elementType, COLLECT(DISTINCT label) AS labels;"""

        all_labels: List[Dict[str, Any]] = self.graph_query(labels_query, params={"EXCLUDED_LABELS": ["MENTIONS", "Node", "DocumentPage"]})
        node_labels = []
        rel_labels = []
        for lab_dict in all_labels:
            if lab_dict['elementType'] == 'node':
                node_labels.extend(lab_dict['labels'])
            elif lab_dict['elementType'] == 'relationship':
                rel_labels.extend(lab_dict['labels'])

        return set(node_labels), set(rel_labels) 

    
    def docs2nodes_and_rels(self, docs: List[Document]) -> Tuple[List[Document], List[Node], List[Relationship]]:
        
        # Only store unique nodes and relationships - Helps later when adding to neo4j
        nodes_dict: Dict[str, list] = dict({})
        rels: list = []
        
        ex_node_types, _ = self.__get_existing_labels()
        # ex_node_types = set({})
        # ex_rel_types = set({})

        for i, doc in enumerate(docs):
            output = None

            if len(doc.page_content) >= 15:
                output: KnowledgeGraph = self.data_ext_chain.invoke(
                    {
                        "subject": self.text_subject,
                        "input": doc.page_content, 
                        "node_types": ','.join(list(ex_node_types))
                    } # 'rel_types': ','.join(list(ex_rel_types))
                )
            
            if output is not None:
                output_nodes: List[Node] = format_nodes(output.nodes)
                output_rels: List[Relationship] = format_rels(output.rels)
                
                if self.verbose:
                    print('-'*100)
                    print(f"Doc # {i+1}")
                    print(f"Nodes: {output_nodes}")
                    print(f"Relationships: {output_rels}\n")

                ntypes = set([n.type for n in output_nodes])
                # rtypes = set([r.type for r in output_rels])
                
                # Store nodes with same type and ID together so that they can be merged together later
                dnodes_dict: Dict[str, List[str]] = {}
                for n in output_nodes:
                    nk = f"{n.id}::{n.type}"
                    if nk in dnodes_dict:
                        if not n in dnodes_dict[nk]:
                            dnodes_dict[nk].append(n)
                    else:
                        dnodes_dict[nk] = [n]
                
                for r in output_rels:
                    if not r in rels:
                        rels.append(r)

                # Store Node IDs+Node Type of nodes mentioned in this doc/chunk in doc metadata
                doc.metadata['mentions'] = list(dnodes_dict.keys())
                nodes_dict = {**nodes_dict, **dnodes_dict}
                # Update existing node and relationship types for next iteration
                ex_node_types: set[str] = ex_node_types.union(ntypes)
                # ex_rel_types: set[str] = ex_rel_types.union(rtypes)
        
        nodes: List[Node] = [merge_nodes(nds) for _, nds in nodes_dict.items()]

        return docs, nodes, list(rels)

    def disambiguate_nodes(
            self, 
            docs: List[Document], 
            nodes: List[Node],
            rels: List[Relationship]
        ) -> Tuple[List[Document], List[Node], List[Relationship]]:

        nnodes, nrels = self.disambiguator.run(nodes, rels)
        return docs, nnodes, nrels

    def process_documents(self, docs: List[Document]):
        docs, nodes, rels = self.docs2nodes_and_rels(docs)
        
        if self.disambiguator is not None:
            docs, nodes, rels = self.disambiguate_nodes(docs, nodes, rels)

        self._upsert_nodes(nodes)
        self._upsert_rels(rels)
        self._create_text_nodes(docs)
