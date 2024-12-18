import os
from typing import List, Dict, Any, Callable, Tuple, Generator

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

from kgrag.entity_linking import link_nodes
from kgrag.prompts import DATA_EXTRACTION_SYSTEM

from loguru import logger

def chunks(lst: List, n: int) -> Generator[list, Any, None]:
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def escape_lucene_special_characters(text) -> str:
    # List of special characters in Lucene query syntax
    special_chars = r'[+\-!(){}[\]^"~*?:\\/]'
    
    # Function to add a backslash before each special character
    return re.sub(special_chars, r'\\\g<0>', text)

def generate_full_text_query(input_str: str) -> str:
    """
    Generate a full-text search query for a given input string.

    This function constructs a query string suitable for a full-text search.
    It processes the input string by splitting it into words and appending a
    similarity threshold (~2 changed characters) to each word, then combines 
    them using the OR operator. Useful for mapping entities from user questions
    to database values, and allows for some misspelings.
    """
    
    input_str = escape_lucene_special_characters(input_str.lower())
    words: List[str] = [el for el in input_str.split() if len(el) > 0]
    if len(words) <= 1:
        return input_str.strip()
    full_text_query: str = ""
    for word in words[:-1]:
        if len(word) <= 4:
            full_text_query += f"{word} AND "
        else: 
            full_text_query += f"{word}~2 AND "
    full_text_query += f"{words[-1]}~2"
    return full_text_query.strip()

def get_extraction_chain(llm: BaseLanguageModel, prompt: str = DATA_EXTRACTION_SYSTEM) -> RunnableSerializable[Dict, Dict | BaseModel]:
    
    if 'gemini-1.0' in llm.model:
        prompt: ChatPromptTemplate = ChatPromptTemplate.from_messages(
        [
            ("human", prompt),
            ("human", "Use the given format to extract information from the following input which is a small sample from a much larger text belonging to the same subject matter: {input}"),
            ("human", "Tip: Make sure to answer in the correct format"),   
        ])
    else:
        prompt = ChatPromptTemplate.from_messages(
        [
            ("system", prompt),
            ("human", "Use the given format to extract information from the following input which is a small sample from a much larger text belonging to the same subject matter: {input}"),
            ("human", "Tip: Make sure to answer in the correct format"),
        ])
    
    return prompt | llm.with_structured_output(KnowledgeGraph)


class Text2KG:

    def __init__(
            self, 
            llm: BaseLanguageModel, 
            emb_model: Embeddings | SentenceTransformer | None = None, 
            data_ext_prompt: str = DATA_EXTRACTION_SYSTEM,
            # disambiguate_nodes: bool = False,
            link_nodes: bool = True,
            **kwargs
        ) -> None:
        """
        The Text 2 Knowledge Graph converter

        Params:
            
            llm (language_core.language_models.BaseLanguageModel): The LLM to use
            
            emb_model (language_core.language_models.Embeddings | sentence_transformers.SentenceTransformer | None): The embeddings model. Default=None.
            
            data_ext_prompt (str): The prompt to use in the data extraction chain. Default=DATA_EXTRACTION_SYSTEM.
            
            link_nodes (bool): Whether to link the nodes or not. Default=True
            
            subject (str | None): Optional. The subject or topic that the documents provided will likely belong to.
                                    Default=None

            neo4j_url (str): The full URL to the Neo4j server. Default="bolt://localhost:7687"

            neo4j_username (str): The username of the Neo4j authenticated user. Default="neo4j"

            neo4j_password (str): The password of the Neo4j server authentication. Default=None
            
            neo4j_database (str): The name of the Neo4j database to use. Default="neo4j".
            
            embed_model_name (str): Only used when emb_model is None. The name of the SentenceTransformer model
                                to use for generating embeddings.Default="sentence-transformers/all-MiniLM-L6-v2"
            
            node_vector_similarity_threshold (float): The minimum vector similarity score to match a new node to an
                                existing node. Default=0.85 (0<=score<=1.0)
            
            node_id_fulltext_similarity_threshold (float): The minimum fulltext similarity score to match the node ID
                                of a node mentioned in a relationship or document to an existing node. Default=1.0
            
            verbose (bool): Whether to log/print the progress of the KG creation process. Default=False
         """
        
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
                "Either set it using the NEO4J_URL environment variable or pass the neo4j_url argument"
            )
        except neo4j.exceptions.AuthError:
            raise ValueError(
                "Could not connect to Neo4j database. "
                "Please ensure that the username and password are correct"
                "Either set the username and password using the NEO4J_USERNAME and NEO4J_PASSWORD environment variables."
                "Or pass the neo4j_username and neo4j_password arguments."
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

        self.data_ext_chain: RunnableSerializable[Dict, Dict | BaseModel] = get_extraction_chain(llm, data_ext_prompt)
        if emb_model is None:
            emb_model = SentenceTransformer(
                kwargs.get("embed_model_name", 'sentence-transformers/all-MiniLM-L6-v2'), 
                cache_folder=os.environ.get("MODELS_CACHE_FOLDER", None),
                tokenizer_kwargs={"clean_up_tokenization_spaces": False}
            )

        self.emb_model: SentenceTransformer | Embeddings = emb_model

        # self._disambiguate: bool = disambiguate_nodes
        self.disambiguator: Callable | None = None
        # if disambiguate_nodes:
        #     from kgrag.data_disambiguation import DataDisambiguation

        #     disambiguator = DataDisambiguation(llm=kwargs.get("disambiguate_llm", None) or llm)
        #     self.disambiguator = disambiguator.run
        
        self.link_nodes = link_nodes

        self.__create_indexes()
    
    def __del__(self):
        if self._driver is not None:
            self._driver.close()

    def __create_indexes(self) -> None:
        ftquery = """CREATE FULLTEXT INDEX IDsAndAliases IF NOT EXISTS FOR (n:Node) ON EACH [n.id, n.alias, n.labels]
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
            "CREATE CONSTRAINT unique_id IF NOT EXISTS FOR (n:Node) REQUIRE n.id IS UNIQUE;",
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
    'MERGE (node: Node {id: $nid}) WITH node CALL db.create.setNodeVectorProperty(node, "embedding", $embedding) RETURN node',
    'RETURN $n AS node',
    {n: n, ntype: nd.type, nid: nd.id, embedding: nd.embedding}
)
YIELD value AS nn
WITH nd AS nd, nn.node AS nn
CALL apoc.create.addLabels(nn, [nd.type]) YIELD node AS nnn
WITH nd, nnn
SET nnn += nd.properties
FOREACH(x in CASE WHEN nnn.alias IS NULL OR NOT (nd.id IN nnn.alias OR nd.id = nnn.id) THEN [1] END | 
    SET nnn.alias = COALESCE(nnn.alias,[]) + nd.id )
RETURN DISTINCT nnn;
"""
        if len(nodes) == 0:
            if self.verbose:
                logger.info("No (unmatched) nodes to upsert")
            return
        # Use Node ID to create a string to generate embeddings that will be used to calculate similarity
        emb_strings: List[str] = []
        for node in nodes:
            s = f"{node.id}, {convert_case(node.type).lower()}"
            if len(node.aliases) > 0:
                s += ', ' + ', '.join(node.aliases)
            if len(node.definition) > 0:
                s += ', ' + node.definition
            emb_strings.append(s.strip(', '))
            # emb_strings.append(f"{node.id} {convert_case(node.type)} {node.definition} {node.aliases}")
        embeddings: List[List[float | double]] = self._embed(emb_strings)
        nodes = [
            {
                "id": node.id,
                "type": node.type,
                "properties": {p.key: p.value for p in node.properties} | {"definition": node.definition, "labels": node.aliases},
                "embedding": list(embeddings[i])
            }
            for i, node in enumerate(nodes)
        ]
        for bnodes in chunks(nodes, self._node_batch_size):
            
            result: List[Dict[str, Any]] = self.graph_query(query, params={"nodes": bnodes, "similarity_threshold": self._sim_thresh})
            # if self.verbose:
            #     logger.info(result)

    def _upsert_matched_nodes(self, nodes: List[Dict[str, str]]):
        '''
        Create new nodes in the Neo4j KG from the given list of nodes

        These nodes are already resolved with unique wikidata IDs so unlike _upsert_nodes,
        this function is much simpler - a simple merge will suffice though embeddings will be created
        and inserted with the nodes here as well if a new node is created
        '''

        query = """UNWIND $nodes AS nd
MERGE (node: Node {id: nd.id})
ON CREATE SET
    node.labels = nd.labels,
    node.url = nd.url,
    node.wiki_description = nd.desc,
    node.definition = nd.definition,
    node.wiki_type = nd.wiki_type
SET node += nd.properties
FOREACH(x in CASE WHEN node.alias IS NULL OR NOT (nd.alias IN node.alias) THEN [1] END | 
    SET node.alias = COALESCE(node.alias,[]) + nd.alias )
WITH node, nd
CALL apoc.create.addLabels(node, [nd.type]) YIELD node AS nn
WITH nn, nd
CALL apoc.do.when(nn.embedding IS NULL,
    "CALL db.create.setNodeVectorProperty($nn, 'embedding', $nd.embedding) RETURN $nn AS node",
    "RETURN $nn AS node",
    {nn: nn, nd: nd}
) YIELD value
WITH value.node AS nne
RETURN nne{.id, .alias, .labels};
"""
        if len(nodes) == 0:
            if self.verbose:
                logger.info("No (matched) nodes to upsert")
            return
        # Use Node ID to create a string to generate embeddings that will be used to calculate similarity
        emb_strings: List[str] = []
        for node in nodes:
            s = f"{node['alias']}, {convert_case(node['type']).lower()}"
            if len(node['labels']) > 0:
                s += ', ' + ', '.join(node['labels'])
            if len(node['wiki_type']) > 0:
                s += ', ' + node['wiki_type']
            if len(node['definition']) > 0:
                s += ', ' + node['definition']
            if len(node['desc']) > 0:
                s += ', ' + node['desc']
            emb_strings.append(s.strip(', '))
        embeddings: List[List[float | double]] = self._embed(emb_strings)
        nodes = [
            {
                "id": node['id'],
                "type": node['type'],
                "desc": node.get("desc", ""),
                "definition": node.get("definition", ""),
                "url": node.get("url", ""),
                "wiki_type": node.get("wiki_type", ""),
                "labels": node.get("labels", []),
                "alias": node["alias"],
                "properties": node.get('properties', {}),
                "embedding": list(embeddings[i])

            }
            for i, node in enumerate(nodes)
        ]
        for cnodes in chunks(nodes, 500):
            result: List[Dict[str, Any]] = self.graph_query(query, params={"nodes": cnodes})
            # if self.verbose:
            #     logger.info(result)

    def _upsert_rels(self, rels: List[Relationship]) -> None:
        '''
        Creates new relationships between existing nodes in the KG
        '''
        query = """
UNWIND $rels AS r
CALL (r) {
    CALL db.index.fulltext.queryNodes("IDsAndAliases", r.start_node_id) YIELD node, score
    WHERE score > $similarity_threshold
    RETURN node AS start_node LIMIT 1
}
CALL (r) {
    CALL db.index.fulltext.queryNodes("IDsAndAliases", r.end_node_id) YIELD node, score
    WHERE score > $similarity_threshold
    RETURN node AS end_node LIMIT 1
}
WITH r, start_node, end_node
WHERE start_node <> end_node
CALL apoc.merge.relationship(start_node, r.type, r.properties, {}, end_node, {}) YIELD rel
RETURN DISTINCT start_node{.id,.alias}, rel, end_node{.id, .alias};
""".strip()
        
        #apoc.text.replace(r.start_node_id, '([+\\-!(){}\\[\\]^"~*?:\\\\/])', '\\\\$1')
        if len(rels) == 0:
            if self.verbose:
                logger.info("No relationships to upsert")
            return
        rels = [
            {
                "start_node_id": generate_full_text_query(rel.start_node_id),
                "end_node_id": generate_full_text_query(rel.end_node_id),
                "type": rel.type,
                "properties": {p.key: p.value for p in rel.properties} | {"context": rel.context}
            }
            for rel in rels
        ]
        for crels in chunks(rels, 500):
            new_rels: List[Dict[str, Any]] = self.graph_query(query, params={"rels": crels, "similarity_threshold": self._ft_sim_thresh})
            # if self.verbose:
            #     logger.info(new_rels)

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
RETURN DISTINCT d.source, collect(node.alias) AS nodes"""

        for doc in docs:
            # Prepare the entity mentions in this particular doc
            mentions: List[str] = doc.metadata.pop("mentions", [])
            mentions = [
                {
                    "id": escape_lucene_special_characters(m.split('::')[0].strip()), 
                    "type": m.split('::')[-1].strip() if len(m.split('::')) > 1 else "Node"
                }
                for m in mentions
            ]

            source: str = doc.metadata.pop("source", '')
            if source is not None or len(source) > 0:
                if not "page" in source.lower() or not "chunk" in source.lower():
                    source = f"{source.split('/')[-1]} Page: {str(doc.metadata.get('page', 0))}"
            elif 'filename' in doc.metadata:
                source = f"{doc.metadata.get('filename', '').split('/')[-1]} Page: {str(doc.metadata.get('page', 0))}"
            elif 'filepath' in doc.metadata:
                source = f"{doc.metadata.get('filepath', '').split('/')[-1]} Page: {str(doc.metadata.get('page', 0))}"
            else:
                for k, v in doc.metadata.items():
                    if 'file' in k or 'source' in k:
                        source = f"{v} Page: {str(doc.metadata.get('page', 0))}"
                        break
            params = {
                "text": doc.page_content,
                "source": source,
                "mentions": mentions, 
                "properties": doc.metadata,
                "similarity_threshold": self._ft_sim_thresh
            }
            result: List[Dict[str, Any]] = self.graph_query(query, params=params)
            # if self.verbose:
            #     logger.info(result)

    
    def __get_existing_node_labels(self) -> Tuple[set, set]:

        labels_query = """CALL apoc.meta.data() YIELD label, elementType
WHERE NOT label IN $EXCLUDED_LABELS
RETURN elementType, COLLECT(DISTINCT label) AS labels;"""

        all_labels: List[Dict[str, Any]] = self.graph_query(labels_query, params={"EXCLUDED_LABELS": ["MENTIONS", "Node", "DocumentPage"]})
        node_labels = []
        rel_labels = []
        for lab_dict in all_labels:
            if lab_dict['elementType'] == 'node':
                node_labels.extend(lab_dict['labels'])

        return set(node_labels)

    
    def docs2nodes_and_rels(self, docs: List[Document], ex_node_types: set) -> Tuple[List[Document], List[Node], List[Relationship]]:
        
        # Only store unique nodes and relationships - Helps later when adding to neo4j
        nodes_dict: Dict[str, list] = dict({})
        rels: list = []

        for i, doc in enumerate(docs):
            output = None

            if len(doc.page_content) >= 15:
                try:
                    output: KnowledgeGraph = self.data_ext_chain.invoke(
                        {
                            "subject": self.text_subject,
                            "input": doc.page_content, 
                            "node_types": ','.join(list(ex_node_types))
                        } # 'rel_types': ','.join(list(ex_rel_types))
                    )
                except Exception as e:
                    logger.error(e)
                    output = None
            
            if output is not None:
                output_nodes: List[Node] = format_nodes(output.nodes)
                output_rels: List[Relationship] = format_rels(output.rels)
                
                if self.verbose:
                    logger.info(f"Doc # {i+1}:- Number of Nodes Extracted: {len(output_nodes)}")
                    logger.info(f"Doc # {i+1}:- Number of Relationships Extracted: {len(output_rels)}")
                    # logger.info(f"Doc # {i+1}\nNodes:\n{output_nodes}\n\nRelationships:\n{output_rels}\n")

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
            nodes: List[Node],
            rels: List[Relationship]
        ) -> Tuple[List[Document], List[Node], List[Relationship]]:

        nnodes, nrels = self.disambiguator(nodes, rels)
        return nnodes, nrels

    def process_documents(self, docs: List[Document], use_existing_node_types: bool = False):
        '''
        The core function that processes the list of Documents and adds the extracted information into a Knowledge Graph
     
        Args:
            docs (list[langchain_core.documents.base.Document]): The list of documents to process
            
            use_existing_node_types (bool): Whether to use the node types already present in the KG in the data extraction prompt. Slightly limits the LLM in the node types of the nodes that it can extract.
        
        Returns:
            None
        
        '''
        
        ex_node_types = set({})
        if use_existing_node_types:
            ex_node_types = self.__get_existing_node_labels()
        
        for i, cdocs in enumerate(chunks(docs, 50)):
            if self.verbose:
                logger.info(f"BATCH {i+1}\n{'='*100}")
            
            cdocs, nodes, rels = self.docs2nodes_and_rels(cdocs, ex_node_types)

            if self.disambiguator is not None:
                nodes, rels = self.disambiguate_nodes(nodes, rels)

            if self.link_nodes:
                matched_nodes, nodes = link_nodes(nodes, self.emb_model, 0.6, verbose=self.verbose)

            self._upsert_matched_nodes(matched_nodes)
            self._upsert_nodes(nodes)
            self._upsert_rels(rels)
            self._create_text_nodes(cdocs)
