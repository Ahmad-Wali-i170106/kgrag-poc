import os
import re
import json
from typing import List, Dict, Any, LiteralString, Tuple

import neo4j
from neo4j.exceptions import CypherSyntaxError, CypherTypeError
from neo4j.graph import Node as GraphNode, Relationship as GraphRelationship
import pandas as pd

from langchain_core.runnables import Runnable
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AnyMessage
from langchain_core.embeddings import Embeddings
from langchain_community.graphs import Neo4jGraph

from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars

from sentence_transformers import SentenceTransformer

from kgrag.data_schema_utils import Entities
from kgrag.prompts import CYPHER_GENERATION_SYSTEM
from kgrag.examples_getter import ExamplesGetter

def escape_lucene_special_characters(text) -> str:
    # List of special characters in Lucene query syntax
    special_chars = r'[+\-!(){}[\]^"~*?:\\/]'
    
    # Function to add a backslash before each special character
    return re.sub(special_chars, r'\\\g<0>', text)

from loguru import logger

def extract_cypher(text: str) -> str:
    """Extract Cypher code from a text.

    Args:
        text: Text to extract Cypher code from.

    Returns:
        Cypher code extracted from the text.
    """
    # The pattern to find Cypher code enclosed in triple backticks
    pattern = r"```(.*?)```"

    # Find all matches in the input text
    matches = re.findall(pattern, text, re.DOTALL)
    matches = matches[0] if matches else text
    matches = matches = re.sub("cypher", '', matches, flags=re.I)
    return matches.strip().strip(': ')


def construct_schema(
    structured_schema: Dict[str, Any],
    include_types: List[str],
    exclude_types: List[str],
) -> str:
    """Filter the schema based on included or excluded types"""

    def filter_func(x: str) -> bool:
        return x in include_types if include_types else x not in exclude_types

    filtered_schema: Dict[str, Any] = {
        "node_props": {
            k: v
            for k, v in structured_schema.get("node_props", {}).items()
            if filter_func(k)
        },
        "rel_props": {
            k: v
            for k, v in structured_schema.get("rel_props", {}).items()
            if filter_func(k)
        },
        "relationships": [
            r
            for r in structured_schema.get("relationships", [])
            if all(filter_func(r[t]) for t in ["start", "end", "type"])
        ],
    }

    # Format node properties
    formatted_node_props = []
    for label, properties in filtered_schema["node_props"].items():
        props_str = ", ".join(
            [f"{prop['property']}: {prop['type']}" for prop in properties]
        )
        formatted_node_props.append(f"{label} {{{props_str}}}")

    # Format relationship properties
    formatted_rel_props = []
    for rel_type, properties in filtered_schema["rel_props"].items():
        props_str = ", ".join(
            [f"{prop['property']}: {prop['type']}" for prop in properties]
        )
        formatted_rel_props.append(f"{rel_type} {{{props_str}}}")

    # Format relationships
    formatted_rels = [
        f"(:{el['start']})-[:{el['type']}]->(:{el['end']})"
        for el in filtered_schema["relationships"]
    ]

    return "\n".join(
        [
            "Node properties are the following:",
            ",".join(formatted_node_props),
            "Relationship properties are the following:",
            ",".join(formatted_rel_props),
            "The relationships are the following:",
            ",".join(formatted_rels),
        ]
    )

def node_to_str(n: GraphNode) -> str:
    
    props = {}
    for pk, pv in n.items():
        if 'embedding' in pk or (isinstance(pv, list) and len(pv) >= 50):
            continue
        props[pk] = pv
    for idp in ["id", "name", "source", "filename", "filepath"]:
        nid = props.pop(idp, None)
        if nid is not None:
            break
    if nid is None:
        node_str = f":{':'.join(list(n.labels))} {props}"
    else:
        node_str = f"{nid}: {':'.join(list(n.labels))} {props}"
    return node_str

def rel_to_str(r: GraphRelationship, include_nodes: bool = False) -> str:
    props = {}
    for pk, pv in r.items():
        if 'embedding' in pk or (isinstance(pv, list) and len(pv) >= 50):
            continue
        props[pk] = pv
    if include_nodes:
        start_node = node_to_str(r.start_node)
        end_node = node_to_str(r.end_node)
        return f"({start_node})-[:{r.type} {props}]->({end_node})"
    return f"[:{r.type} {props}]"

def expand_and_sanitize(df: pd.DataFrame, include_nodes_in_rels: bool = False) -> pd.DataFrame:
    if len(df.index) == 0:
        return df
    for c in df.columns:
        if isinstance(df.loc[0,c], GraphNode):
            df[c] = df[c].apply(node_to_str)
        elif isinstance(df.loc[0,c], GraphRelationship):
            df[c] = df[c].apply(rel_to_str, include_nodes=include_nodes_in_rels)
        if 'embedding' in c or (isinstance(df.loc[0,c], list) and len(df.loc[0,c]) > 64):
            del df[c]
    return df

class DaiNeo4jGraph(Neo4jGraph):
    '''
    Modify the query method of DaiNeo4jGraph to
    
    1) Convert results to DataFrame with expand=False
    2) Apply expand_and_sanitize method on df which expands neo4j.Node & neo4j.Relationship & removes embedding property
    3) Convert the df to a list of dictionaries & return
    '''
    def query(self, query: str, params: Dict[str, Any] = {}, include_nodes_in_rels: bool = False) -> List[Dict[str, Any]]:
        
        with self._driver.session(database=self._database) as session:
            try:
                data: neo4j.Result = session.run(neo4j.Query(text=query, timeout=self.timeout), params)
                df: pd.DataFrame = data.to_df(expand=False)
                if self.sanitize:
                    df = expand_and_sanitize(df, include_nodes_in_rels)
                return df.to_dict(orient="records")
                
            except (CypherSyntaxError, CypherTypeError) as e:
                raise ValueError(f"Generated Cypher Statement is not valid\n{e}")

# rels_query: LiteralString = """
# CALL (n) {
#     MATCH (n: Node)-[r:!MENTIONS]->(target: Node)
#     RETURN DISTINCT type(r) AS r, target AS m
#     UNION
#     MATCH (n: Node)<-[:MENTIONS]-(:DocumentPage)-[:MENTIONS]->(target: Node)
#     RETURN DISTINCT "MENTIONED_WITH" AS r, target AS m
# }
# WITH n, m, r
# CALL (n) {
#     WITH [p IN keys(n) WHERE NOT p IN ["embedding", "id"] AND n[p] IS :: INTEGER|FLOAT|BOOLEAN|STRING|POINT | p] AS filteredProps, n
#     WITH reduce(node_str = "{" , p in filteredProps | node_str  + toString(p) + ": " + toString(n[p]) + ", ") AS node_str, n
#     WITH reduce(node_label_str = "", l in LABELS(n) | node_label_str + l + ", ") AS node_label_str, node_str, n
#     RETURN n.id + ": " + rtrim(node_label_str, ', ') + " " + RTRIM(node_str, ', ') + "}" AS node_str
# }
# CALL (m) {
#     WITH [p IN keys(m) WHERE NOT p IN ["embedding", "id"] AND m[p] IS :: INTEGER|FLOAT|BOOLEAN|STRING|POINT | p] AS filteredProps, m
#     WITH reduce(tar_str = "{", p in filteredProps | tar_str  + toString(p) + ": " + toString(m[p]) + ", ") AS tar_str, m
#     WITH reduce(tar_label_str = "", l in LABELS(m) | tar_label_str + l + ", ") AS tar_label_str, tar_str, m
#     RETURN m.id + ": " + rtrim(tar_label_str, ', ') + " " + rtrim(tar_str, ', ') + "}" AS tar_str
# }
# WITH node_str, tar_str, r
# RETURN DISTINCT "(" + node_str + ")-[:" + r + "]->(" + tar_str + ")" AS output_string
# LIMIT $limit;""".strip()

rels_query: LiteralString = """
MATCH (n: Node)-[r:!MENTIONS]-(m: Node)
RETURN DISTINCT n, m, type(r) AS r
""".strip()

docs_query: LiteralString = """
MATCH (n: Node)<-[:MENTIONS]-(d:DocumentPage)
RETURN DISTINCT d.source AS document_source, d.text AS document_text, d.chunk_id AS chunk_id
""".strip()

fulltext_search_cypher: LiteralString = """
CALL db.index.fulltext.queryNodes(
    "IDsAndAliases",
    $search_term
) YIELD node AS n, score
WHERE score > $min_score
RETURN DISTINCT n.id AS n, score LIMIT $top_k;
""".strip()

vector_search_cypher: LiteralString = """
CALL db.index.vector.queryNodes(
    "IDsVectors",
    $top_k,
    $embedding
) YIELD node AS n, score
WHERE score > $min_score
RETURN DISTINCT n.id AS n, score LIMIT $top_k;
""".strip()

class KGSearch:
    
    def __init__(
            self, 
            ent_llm: BaseLanguageModel, 
            cypher_llm: BaseLanguageModel, 
            sim_model: SentenceTransformer | Embeddings | None = None,
            cypher_examples_json: str | None = None, 
            **kwargs
        ):
        
        '''
        Args:
            ent_llm (langchain_core.language_models.BaseLanguageModel): The Language Model to use to extract entities
            cypher_llm (langchain_core.language_models.BaseLanguageModel): The language model to use to generate cypher
            sim_model (sentence_transformers.SentenceTransformer | langchain_core.embeddings.Embeddings): 
                        The embedding model to use to generate the embeddings for calculating vector similarity. 
                        Should be same as the one used  in the KG creation process. Default=None
            cypher_examples_json (str): The path to the JSON file containing a list of cypher examples in the format
                                        {"question": <query>, "cypher": <cypher>}
            neo4j_url (str): The full url to the Neo4j server. If NEO4J_URL environment variable is set, uses that instead.
                            Default="bolt://localhost:7687"
            neo4j_username (str): The username to provide for authentication to the Neo4j server.
                                  If NEO4J_USERNAME env variable is set, uses that instead. Default="neo4j"
            neo4j_password (str): The password to provide for authentication to the Neo4j server.
                                  If NEO4J_PASSWORD env variable is set, uses that instead. Default=None
            neo4j_database (str): The neo4j database name to query. If NEO4J_DATABASE env variable is set, uses that instead. Default="neo4j"
            
            fulltext_search_max_difference (float): The maximum score difference allowed between 2 consecutive
                                fulltext search results. Default=2.0. 
                                Use to make sure that search results are always close to each other.
            fulltext_search_min_score (float): The minimum score of a fulltext search result for it to be considered.
                                            Default=1.0
            fulltext_search_top_k (int): The top 'k' fulltext search results to consider. Default=10
            vector_search_top_k (int): The top 'k' vector search results to consider. Default=15
            vector_search_min_score (float): The minimum score of vector search result for it to be considered.
                                            Default=0.75 where 0 <= score <= 1.0.
            max_cypher_fewshot_examples (int): The maximum number of similar cypher fewshot examples given in 
                                                `cypher_examples_json` to choose dynamically. Default=15
            
            embed_model_name (str): Only used when sim_model is None. The name of the SentenceTransformer model
                                to use for generating embeddings.Default="sentence-transformers/all-MiniLM-L6-v2"
        '''

        url: str = os.environ.get("NEO4J_URL", kwargs.get("neo4j_url", "bolt://localhost:7687"))
        username: str = os.environ.get("NEO4J_USERNAME", kwargs.get("neo4j_username", "neo4j"))
        password: str = os.environ.get("NEO4J_PASSWORD", kwargs.get("neo4j_password", None))
        database: str = os.environ.get("NEO4J_DATABASE", kwargs.get("neo4j_database", "neo4j"))
        self.graph = DaiNeo4jGraph(
            url=url, username=username, password=password, database=database,
            enhanced_schema=False, refresh_schema=True, sanitize=True
        )
        if 'gemini-1.0' in ent_llm.model:
            role = "human"
        else:
            role = "system"

        ent_prompt: ChatPromptTemplate = ChatPromptTemplate.from_messages(
            [
                (
                    role,
                    "You are extracting various types of entities from the text.",
                ),
                (
                    "human",
                    "Use the given format to extract information from the following input: {question}",
                ),
            ]
        )

        self.entity_chain: Runnable = ent_prompt | ent_llm.with_structured_output(Entities)
        
        cypher_prompt: ChatPromptTemplate = ChatPromptTemplate.from_messages(
            [
                (
                    role,
                    CYPHER_GENERATION_SYSTEM
                ),
                (
                    "human",
                    "The question is:\n{question}"
                )
            ]
        )

        self.cypher_generation_chain: Runnable = cypher_prompt | cypher_llm

        self.max_difference: float = kwargs.get("fulltext_search_max_difference", 2.0)
        self.ft_min_score: float = kwargs.get("fulltext_search_min_score", 1.0)
        self.ft_top_k: int = kwargs.get("fulltext_search_top_k", 10)
        self.vec_top_k: int = kwargs.get("vector_search_top_k", 15)
        self.vec_min_score: float = kwargs.get("vector_search_min_score", 0.75)
        self.max_examples: int = kwargs.get("max_cypher_fewshot_examples", 15)

        self.sim_model: SentenceTransformer | Embeddings = sim_model
        if self.sim_model is None:
            self.sim_model = SentenceTransformer(
                kwargs.get("embed_model_name", 'sentence-transformers/all-MiniLM-L6-v2'), 
                cache_folder=os.environ.get("MODELS_CACHE_FOLDER", None),
                tokenizer_kwargs={"clean_up_tokenization_spaces": False}
            )
        
        
        self.example_getter = None
        if cypher_examples_json is None or len(cypher_examples_json) == 0:
            self.example_getter = ExamplesGetter(sim_model=self.sim_model, json_filename=cypher_examples_json)

        self._include_types: List[str] = kwargs.get("include_node_types", [])
        self._exclude_types: List[str] = kwargs.get("exclude_node_types", [])
        if len(self._include_types) > 0 and len(self._exclude_types) > 0:
            logger.warning("You can provide either of `include_node_types` or `exclude_node_types`.",
                  "By default, `exclude_node_types` has been ignored (set to empty list []).")
            self._exclude_types = []
        
    def _embed(self, sents: List[str]) -> List[List[float]]:
        if isinstance(self.sim_model, SentenceTransformer):
            return [list(sent) for sent in self.sim_model.encode(sents)]
        elif isinstance(self.sim_model, Embeddings):
            return self.sim_model.embed_documents(sents)

    @staticmethod
    def generate_full_text_query(input_str: str) -> str:
        """
        Generate a full-text search query for a given input string.

        This function constructs a query string suitable for a full-text search.
        It processes the input string by splitting it into words and appending a
        similarity threshold (~2 changed characters) to each word, then combines 
        them using the OR operator. Useful for mapping entities from user questions
        to database values, and allows for some misspelings.
        """
        
        input_str = escape_lucene_special_characters(remove_lucene_chars(input_str.lower()))
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

    def retrieve_node_ids_fulltext(self, entities) -> List[str]:
        node_ids = []
        for ent in entities:
            output: List[Dict[str, str | float]] = self.graph.query(
                fulltext_search_cypher, 
                params={
                    "search_term": self.generate_full_text_query(ent),
                    "min_score": self.ft_min_score,
                    "top_k": self.ft_top_k
                }
            )
            for i, o in enumerate(output):
                if i == 0:
                    node_ids.append(o['n'])
                else:
                    diff: float = output[i-1]['score'] - o['score']
                    if diff >= self.max_difference:
                        break
                    node_ids.append(o['n'])
        return node_ids
    
    def retrieve_node_ids_vector(self, query: str) -> List[str]:
        embedding = self._embed([query])[0]

        output: List[Dict[str, str | float]] = self.graph.query(
            vector_search_cypher,
            params={
                "embedding": embedding,
                "min_score": self.vec_min_score,
                "top_k": self.vec_top_k
            }
        )
        node_ids = [o['n'] for o in output]
        return node_ids
    
    def retrieve_custom_cypher(self, query: str, nresults: int) -> List[str]:
        if self.example_getter is None:
            examples = ""
        else:
            examples: List[str] = self.example_getter.get_examples(query, top_k=self.max_examples, sim_cutoff=0.1)
            examples: str = '\n'.join(examples)
        
        if len(examples) > 0:
            examples = f"""Examples: Here are a few examples of generated Cypher statements for particular questions:

{examples}"""
        else:
            examples = ""
        graph_schema = construct_schema(
            self.graph.get_structured_schema, self._include_types, self._exclude_types
        )
        cypher: str | AnyMessage = self.cypher_generation_chain.invoke(
            {
                "schema": graph_schema,
                "examples": examples,
                "question": query,
            }
        )
        if isinstance(cypher, AnyMessage):
            cypher = cypher.content
        
        cypher = extract_cypher(cypher)
        logger.info(f"Following Cypher was generated:\n{cypher}")
        if cypher:
            try:
                result: List[Dict[str, Any]] = self.graph.query(cypher)[:nresults]
                result = [json.dumps(res) for res in result]
            except ValueError as ve:
                logger.error(ve)
                result = []
            except Exception as e:
                logger.error(e)
                result = []
        else:
            result = []
        return result
    
    def retrieve(
        self, 
        query: str, 
        nresults: int | None = None, 
        use_fulltext_search: bool = True, 
        use_vector_search: bool = True,
        generate_cypher: bool = False,
        return_chunk_ids: bool = True
    ) -> Tuple[List[str], List[str | int], List[str]]:
        '''
        Args:
            
            query (str): The search query
            
            nresults (int): The number of results to return. Default=None (Return all results)
            
            use_fulltext_search (bool): Whether to use fulltext search to search for nodes. Default=True
            
            use_vector_search (bool): Whether to use vector search to search for nodes. Default=True
            
            generate_cypher (bool): Whether to generate cypher from user query to get results. Default=False
            
            return_chunk_ids (bool): Whether to return the chunk IDs of the chunks/docs that are found. Otherwise, return the text content of the docs/chunks
            
        Returns:
            
            rels: All the triples that the entities/nodes in the query are involved in
                 Empty list is returned if `use_fulltext_search=False` && `use_vector_search=False`
     
            docs: All the documents that contain any entities mentioned in the query
                 Empty list is returned if `use_fulltext_search=False` && `use_vector_search=False`
                 If return_chunk_ids=True, returns chunk_ids as set during text2kg conversion / data insertion process
            gen_cypher_results: List of string/JSON results from the generated cypher
                                 Empty list is returned if `generate_cypher=False
        '''
        docs, rels, gen_cypher_results = [], [], []

        if use_fulltext_search or use_vector_search:
            node_ids = set({})
            if use_fulltext_search:
                entities: Entities = self.entity_chain.invoke({"question": query})
                if entities is not None:
                    entities: List[str] = entities.names
                    if len(entities) > 0:
                        node_ids.update(set(self.retrieve_node_ids_fulltext(entities)))
            
            if use_vector_search and self.sim_model is not None:
                node_ids.update(set(self.retrieve_node_ids_vector(query)))
            
            if len(node_ids) == 0:
                logger.info("No Nodes were found using Fulltext/Vector Search",
                      "Generating cypher to retrieve any/all results")
                gen_cypher_results = self.retrieve_custom_cypher(query, nresults)
                return [], [], gen_cypher_results
            else:
                if nresults is not None:
                    rels = self.graph.query(
                        f"UNWIND $node_ids AS nid\nMATCH (n: Node {{id: nid}})\n{rels_query}\nLIMIT $limit", 
                        {"node_ids": list(node_ids), "limit": max(int(nresults), 1)},
                        include_nodes_in_rels=False
                    )

                    docs = self.graph.query(
                        f"UNWIND $node_ids AS nid\nMATCH (n: Node {{id: nid}})\n{docs_query}\nLIMIT $limit", 
                        {"node_ids": list(node_ids), "limit": max(int(nresults), 1)}
                    )
                else:
                    rels = self.graph.query(
                        f"UNWIND $node_ids AS nid\nMATCH (n: Node {{id: nid}})\n{rels_query}", 
                        {"node_ids": list(node_ids)},
                        include_nodes_in_rels=False
                    )

                    docs = self.graph.query(
                        f"UNWIND $node_ids AS nid\nMATCH (n: Node {{id: nid}})\n{docs_query}", 
                        {"node_ids": list(node_ids)}
                    )
                # rels: str = '\n'.join([rel['output_string'] for rel in rels])
                if len(rels) > 0:
                    rels: List[str] = [f"({rel['n']})-[:{rel['r']}]->({rel['m']})" for rel in rels]
                else:
                    rels = self.graph.query("UNWIND $node_ids AS nid\nMATCH (n: Node {id: nid})\nReturn n;", {"node_id": list(node_ids)})
                    rels: List[str] = [rel['n'] for rel in rels]
                if not return_chunk_ids:
                    docs: List[str] = [f"{doc['document_text']}\nSOURCE: {doc['document_source']}" for doc in docs]
                else:
                    docs: List[int] = [int(doc['chunk_id']) for doc in docs]
        
        if generate_cypher:
            gen_cypher_results = self.retrieve_custom_cypher(query, nresults)
            # output_string += '\n' + self.retrieve_custom_cypher(query, nresults)
        
        return rels, docs, gen_cypher_results
    
    def retrieve_as_string(
        self,
        query: str, 
        nresults: int | None = None, 
        use_fulltext_search: bool = True, 
        use_vector_search: bool = False,
        generate_cypher: bool = False
    ) -> str:
        rels, docs, gen_cypher_results = self.retrieve(
            query, 
            nresults, 
            use_fulltext_search, 
            use_vector_search,
            generate_cypher,
            return_chunk_ids=False
        )

        output_string = ""
        if len(rels) > 0:
            output_string += f"Nodes Relations:-\n{'\n'.join(rels)}"
        if len(docs) > 0:
            output_string += f"\n\nDocuments:-\n{'\n=====\n'.join(docs)}"
        if len(gen_cypher_results) > 0:
            output_string += f"\n\nCypher Results:-\n{'\n'.join(gen_cypher_results)}"
        
        return output_string.strip()
