import os
from typing import List, Dict, Any, Optional, Tuple


from sentence_transformers import SentenceTransformer
from langchain.pydantic_v1 import Field, BaseModel

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseLanguageModel

from langchain_community.graphs.neo4j_graph import Neo4jGraph
# from langchain_google_genai import ChatGoogleGenerativeAI

SYSTEM_PROMPT = """# Knowledge Graph Instructions for GPT-4
## 1. Overview
You are a top-tier algorithm designed for extracting information in structured formats to build a knowledge graph.
You need to extract the information from consecutive samples of text from a much larger text, most likely a book or short tutorial.
Only extract information that seems most relevant to the overall subject theme of the text.
- **Nodes** represent entities and concepts. They're akin to Wikipedia nodes.
- **Relationships** represent the relations or edges between these **nodes**.
- The aim is to achieve simplicity and clarity in the knowledge graph, making it accessible for a vast audience.
## 2. Labeling Nodes
- **Node IDs**: Never utilize integers as node IDs. Node IDs should be names or human-readable identifiers found in the text. Node IDs should be clear and unambiguous. Do NOT use abbreviations for Node IDs e.g. use Artificial Intelligence instead of AI.
- **Node IDs Naming Convention**: Node IDs should always be written in Title Case e.g., `John Doe`.
- **Node Type Naming Convention**: Node types should always be written in PascalCase e.g., `FictionBook`.
- **Suggested Node Types:**: Following are some suggested node types that you can use: {node_types}
- - Remember that you are NOT limited to these node types. Always choose the node type to be as specific as possible e.g. choose Artist instead of Person if the subject performs some art.
## 3. Labeling Relationships
- **Relationship Type Naming Convention**: Relationship types should be written in SCREAMING_SNAKE_CASE e.g., `RELATED_TO`.
- **Suggested Relationship Types**: Following are some suggested relationship types/labels that you can use: {rel_types}
- - Remember that you are NOT limited to these relationship types.
## 4. Handling Numerical Data and Dates
- Numerical data, like age or other related information, should be incorporated as attributes or properties of the respective nodes.
- **No Separate Nodes for Dates/Numbers**: Do not create separate nodes for dates or numerical values. Always attach them as attributes or properties of nodes.
- **Property Format**: Properties must be in a key-value format.
- **Quotation Marks**: Never use escaped single or double quotes within property values.
- **Naming Convention**: Use camelCase for property keys, e.g., `birthDate`.
- **Type Property**: NEVER include the `type` property in the list of properties.
## 5. Coreference Resolution
- **Maintain Entity Consistency**: When extracting entities, it's vital to ensure consistency.
- If an entity, such as "John Doe", is mentioned multiple times in the text but is referred to by different names or pronouns (e.g., "Joe", "he"), 
always use the most complete identifier for that entity throughout the knowledge graph. In this example, use "John Doe" as the entity ID.  
Remember, the knowledge graph should be coherent and easily understandable, so maintaining consistency in entity references is crucial. 
## 6. Strict Compliance
Adhere to the rules strictly. Non-compliance will result in termination."""


class Property(BaseModel):
    """A single property consisting of key and value"""
    key: str = Field(..., description="key")
    value: str = Field(..., description="value")

class Node(BaseModel):
    id: str = Field(description="The identifying property of the node")
    type: str = Field(description="The entity type / label of the node. It should be as specific as possible.")
    properties: Optional[List[Property]] = Field(
        None, description="List of node properties as a list of key-value pairs.")

class Relationship(BaseModel):
    start_node_id: str = Field(description="The id of the first node in the relationship")
    end_node_id: str = Field(description="The id of the second node in the relationship")
    type: str = Field(description="The label / type / name of the relationship. It should be specific but also timeless.")
    properties: Optional[List[Property]] = Field(
        None, description="List of relationship properties as a list of key-value pairs."
    )
        
class KnowledgeGraph(BaseModel):
    """Generate a knowledge graph with entities and relationships."""
    nodes: list[Node] = Field(
        ..., description="List of nodes in the knowledge graph")
    rels: list[Relationship] = Field(
        ..., description="List of relationships in the knowledge graph"
    )


def _format_nodes(nodes: List[Node]):
    return [
        Node(
            id = n.id.title() if isinstance(n.id, str) else n.id,
            type = n.type.title().replace(' ','').replace('_',''),
            properties = [Property(p.key, p.value) for p in n.properties if p.key.lower() != 'type']
        )
        for n in nodes
    ]

def _format_rels(rels: List[Relationship]):
    return [
        Relationship(
            start_node_id = r.start_node_id.title() if isinstance(r.start_node_id, str) else r.start_node_id,
            end_node_id = r.end_node_id.title() if isinstance(r.end_node_id, str) else r.start_node_id,
            type = r.type.title().replace(' ','').replace('_',''),
            properties = r.properties
        )
        for r in rels
    ]

def _merge_nodes(nodes: List[Node]) -> Node:
    '''
    Merges all the nodes with the same ID and type into a single node

    The only difference in these nodes is their list of properties so 
    I extract and combine a list of unique properties from all these nodes

    Args: 
        nodes (List[Node]):- List of nodes to merge into one
    
    Returns: A single Node object created by merging all the nodes
    '''
    props: List[Property] = []
    for node in nodes:
        props.extend([
            p for p in node.properties
            if not p in props
        ])
    return Node(id=nodes[0].id, type=nodes[0].type, properties=props)


def get_extraction_chain(llm: BaseLanguageModel):
    prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "Use the given format to extract information from the following input which is a small sample from a much larger text belonging to the same subject matter: {input}"),
        ("human", "Tip: Make sure to answer in the correct format"),
    ])
    
    return prompt | llm.with_structured_output(KnowledgeGraph)


class Text2KG:

    def __init__(self, llm: BaseLanguageModel, **kwargs):

        self.graph = Neo4jGraph(
            url=os.environ.get("NEO4J_URL", kwargs.get("NEO4J_URL", "localhost")),
            username=os.environ.get("NEO4J_USERNAME", kwargs.get("NEO4J_USERNAME", "neo4j")),
            password=os.environ.get("NEO4J_PASSWORD", kwargs.get("NEO4J_PASSWORD", None)),
            database=os.environ.get("NEO4J_DATABASE", kwargs.get("NEO4J_DATABASE", "neo4j")),
            sanitize=True,
            refresh_schema=True
        )

        self.verbose = kwargs.get("verbose", False)
        self._sim_thresh = kwargs.get("node_vector_similarity_threshold", 0.85)
        self._ft_sim_thresh = kwargs.get("node_id_fulltext_similarity_threshold", 5)
        
        self.data_ext_chain = get_extraction_chain(llm)
        self.emb_model = SentenceTransformer(kwargs.get("embed_model_name", 'sentence-transformers/all-MiniLM-L6-v2'))

        self.__create_indexes()

    def __create_indexes(self):
        query = """CREATE FULLTEXT INDEX IDsAndAliases IF NOT EXISTS FOR (n:Node) ON EACH [n.id, n.alias];
CREATE VECTOR INDEX IDsVectors IF NOT EXISTS FOR (n:Node) ON n.embedding
OPTIONS { index_config: {
    `vector.dimensions`: 384,
    `vector.similarity_function`: 'cosine'
} };
"""
        self.graph.query(query) # Create the fulltext and vector indexes

    def _embed(self, sents: List[str]):
        return self.emb_model.encode(sents)

    def _upsert_nodes(self, nodes: List[Node]):
        '''
        Create new nodes in the Neo4j KG from the given list of nodes

        Performs an additional step of resolving and merging each new node to an existing node 
        if it exceeds a certain similarity threshold with an existing node.
        The ID of the new node is added to an alias property of the existing node.
        If no match with an existing node is found, a new node is created.

        In affect, it is a more complicated form of MERGE clause that uses semantic similarity
        to only create nodes that are considered semantically different.
        '''
        query = """UNWIND $nodes AS node
CALL db.index.vector.queryNodes('IDsVectors', 1, node.embedding) YIELD node AS n, score
WHERE score > $similarity_threshold AND apoc.label.exists(n, node.type)
CALL apoc.do.when(n IS NULL,
    'CALL apoc.create.node(["Node", node.type], {id: node.id}) YIELD node RETURN node AS n',
    'FOREACH(x in CASE WHEN n.alias IS NULL OR NOT (node.id IN n.alias) THEN [1] END | SET n.alias = COALESCE(n.alias,[]) + node.id ) RETURN n',
    {}
)
SET n += node.properties
CALL db.create.setNodeVectorProperty(n, 'embedding', node.embedding)
RETURN n;"""
        
        embeddings = self.emb_model.encode([node.id for node in nodes])
        nodes = [
            {
                "id": node.id,
                "type": node.type,
                "properties": {p.key: p.value for p in node.properties},
                "embedding": embeddings[i]

            }
            for i, node in enumerate(nodes)
        ]
        new_nodes = self.graph.query(query, params={"nodes": nodes, "similarity_threshold": self._sim_thresh})
        if self.verbose:
            print(new_nodes)

    def _upsert_rels(self, rels: List[Relationship]):
        '''
        Creates new relationships between existing nodes in the KG
        '''
        query = """UNWIND $rels AS rel
CALL {
    CALL db.index.fulltext.queryNodes('IDsAndAliases', rel.start_node_id) YIELD node, score
    WHERE score > $similarity_threshold
    RETURN node AS start_node
}
CALL {
    CALL db.index.fulltext.queryNodes('IDsAndAliases', rel.end_node_id) YIELD node, score
    WHERE score > $similarity_threshold
    RETURN node AS end_node
}
CALL apoc.merge.relationship(start_node, rel.type, rel.properties, {}, end_node, {}) YIELD rel
RETURN rel;"""

        rels = [
            {
                "start_node_id": rel.start_node_id,
                "end_node_id": rel.end_node_id,
                "type": rel.type,
                "properties": {p.key: p.value for p in rel.properties}
            }
            for rel in rels
        ]
        new_rels = self.graph.query(query, params={"rels": rels, "similarity_threshold": self._ft_sim_thresh})
        if self.verbose:
            print(new_rels)

    def _create_text_nodes(self, docs: List[Document]):
        '''
        Creates new Text nodes also called DocumentPage nodes
        '''
        cquery = """MERGE (d: DocumentPage {source: $source})
SET
    d = $properties,
    d.source = $source,
    d.text = $text
RETURN d;"""

        mentions_query = """UNWIND $mentions AS mention_id
CALL db.index.fulltext.queryNodes('IDsAndAliases', mention_id) YIELD node, score
WHERE score > $similarity_threshold
MATCH (d: DocumentPage {source: $source})
MERGE (d)-[:MENTIONS]->(node);"""
        for doc in docs:
            mentions = doc.metadata.pop("mentions", [])
            source = doc.metadata.pop("source", doc.metadata.get("filename", "") + " Page: " + str(doc.metadata.get("page", 0)))
            params = {
                "text": doc.page_content,
                "source": source,
                "properties": doc.metadata
            }
            new_doc_node = self.graph.query(cquery, params=params)
            if self.verbose:
                print(new_doc_node)
            params = {
                "source": source, 
                "mentions": mentions, 
                "similarity_threshold": self._ft_sim_thresh
            }
            self.graph.query(mentions_query, params=params)

    
    def __get_existing_labels(self) -> Tuple[set, set]:
        node_props: Dict[str, Any] = self.graph.structured_schema['node_props']
        rel_props: Dict = self.graph.structured_schema['rel_props']
        return set(node_props.keys()), set(rel_props.keys())

    
    def docs2nodes_and_rels(self, docs: List[Document]) -> Tuple[List[Document], List[Node], List[Relationship]]:
        
        # Only store unique nodes and relationships - Helps later when adding to neo4j
        nodes_dict: Dict[str, set] = dict({})
        rels = set({})
        
        ex_node_types, ex_rel_types = self.__get_existing_labels()
        # ex_node_types = set({})
        # ex_rel_types = set({})

        for i, doc in enumerate(docs):
            output: KnowledgeGraph = self.data_ext_chain.invoke(
                {
                    "input": doc.page_content, 
                    "node_types": ','.join(list(ex_node_types)), 
                    'rel_types': ','.join(list(ex_rel_types))
                }
            )
            if output is not None:
                if self.verbose:
                    print('-'*100)
                    print(f"Doc # {i+1}")
                    print(f"Nodes: {output.nodes}")
                    print(f"Relationships: {output.rels}\n")
                
                output_nodes = _format_nodes(output.nodes)
                output_rels = _format_rels(output.rels)

                ntypes = set([n.type for n in output_nodes])
                rtypes = set([r.type for r in output_rels])
                
                # Store nodes with same type and ID together so that they can be merged together later
                for n in output_nodes:
                    nk = f"{n.id}:{n.type}"
                    if n.id in nodes:
                        nodes_dict[nk].add(n)
                    else:
                        nodes_dict[nk] = set({n})
                
                for r in output_rels:
                    rels.add(r)

                # Store Node IDs+Node Type of nodes mentioned in this doc/chunk in doc metadata
                doc.metadata['mentions'] = [list(nodes_dict.keys())] 
                
                # Update existing node and relationship types for next iteration
                ex_node_types = ex_node_types.union(ntypes)
                ex_rel_types = ex_rel_types.union(rtypes)
        
        nodes: List[Node] = [_merge_nodes(nds) for nid, nds in nodes_dict.items()]

        return docs, nodes, list(rels)

    def text2kg(self, docs: List[Document]):
        docs, nodes, rels = self.docs2nodes_and_rels(docs)
        
        self._upsert_nodes(nodes)
        self._upsert_rels(rels)
        self._create_text_nodes(docs)
