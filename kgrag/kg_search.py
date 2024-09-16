import os
import re
from typing import List, Dict, Any, LiteralString, Optional, Tuple, Generator

import neo4j
from neo4j.exceptions import CypherSyntaxError, CypherTypeError
from neo4j.graph import Node as GraphNode, Relationship as GraphRelationship
import pandas as pd

from langchain_core.runnables import Runnable
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate

from langchain_community.graphs import Neo4jGraph

from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars

from kgrag.data_schema_utils import Entities

def get_examples(query: str, max_examples: int = 10) -> List[str]:
    '''
    Gets `max_examples` or less cypher examples that are most similar to the input query
    '''
    pass

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

    return matches[0] if matches else text


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
        node_str = f"{props} : {', '.join(list(n.labels))}"
    else:
        node_str = f"{nid}: {', '.join(list(n.labels))} {props}"
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
    


rels_query: LiteralString = """
CALL (n) {
    MATCH (n: Node)-[r:!MENTIONS]->(target: Node)
    RETURN DISTINCT type(r) AS r, target AS m
    UNION
    MATCH (n: Node)<-[:MENTIONS]-(:DocumentPage)-[:MENTIONS]->(target: Node)
    RETURN DISTINCT "MENTIONED_WITH" AS r, target AS m
}
WITH n, m, r
CALL (n) {
    WITH [p IN keys(n) WHERE NOT p IN ["embedding", "id"] AND n[p] IS :: INTEGER|FLOAT|BOOLEAN|STRING|POINT | p] AS filteredProps, n
    WITH reduce(node_str = "{" , p in filteredProps | node_str  + toString(p) + ": " + toString(n[p]) + ", ") AS node_str, n
    WITH reduce(node_label_str = "", l in LABELS(n) | node_label_str + l + ", ") AS node_label_str, node_str, n
    RETURN n.id + ": " + rtrim(node_label_str, ', ') + " " + RTRIM(node_str, ', ') + "}" AS node_str
}
CALL (m) {
    WITH [p IN keys(m) WHERE NOT p IN ["embedding", "id"] AND m[p] IS :: INTEGER|FLOAT|BOOLEAN|STRING|POINT | p] AS filteredProps, m
    WITH reduce(tar_str = "{", p in filteredProps | tar_str  + toString(p) + ": " + toString(m[p]) + ", ") AS tar_str, m
    WITH reduce(tar_label_str = "", l in LABELS(m) | tar_label_str + l + ", ") AS tar_label_str, tar_str, m
    RETURN m.id + ": " + rtrim(tar_label_str, ', ') + " " + rtrim(tar_str, ', ') + "}" AS tar_str
}
WITH node_str, tar_str, r
RETURN DISTINCT "(" + node_str + ")-[:" + r + "]->(" + tar_str + ")" AS output_string;
LIMIT $limit""".strip()

docs_query: LiteralString = """
MATCH (n: Node)<-[:MENTIONS]-(d:DocumentPage)
RETURN n.id AS entity, d.source AS document_source, d.text AS document_text
LIMIT $limit""".strip()

fulltext_search_cypher: LiteralString = """
CALL db.index.fulltext.queryNodes(
    "IDsAndAliases",
    $search_term
) YIELD node AS n, score
WHERE score > $min_score
RETURN n.id AS n, score LIMIT $top_k;
""".strip()

class KGSearch:
    
    def __init__(self, ent_llm: BaseLanguageModel, cypher_llm: BaseLanguageModel, **kwargs):

        url: str = os.environ.get("NEO4J_URL", kwargs.get("neo4j_url", "bolt://localhost:7687"))
        username: str = os.environ.get("NEO4J_USERNAME", kwargs.get("neo4j_username", "neo4j"))
        password: str = os.environ.get("NEO4J_PASSWORD", kwargs.get("neo4j_password", None))
        database: str = os.environ.get("NEO4J_DATABASE", kwargs.get("neo4j_database", "neo4j"))
        self.graph = DaiNeo4jGraph(
            url=url, username=username, password=password, database=database,
            enhanced_schema=False, refresh_schema=True, sanitize=True
        )

        ent_prompt: ChatPromptTemplate = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
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
                    "system",
                    """Task:Generate Cypher statement to query a graph database.
Instructions:
Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.
Schema:
{schema}
Note: Do not include any explanations or apologies in your responses.
Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.
Do not include any text except the generated Cypher statement.
Examples: Here are a few examples of generated Cypher statements for particular questions:

{examples}
"""
                ),
                (
                    "human",
                    "The question is:\n{question}"
                )
            ]
        )

        self.cypher_generation_chain: Runnable = cypher_prompt | cypher_llm

        self.max_difference: float = kwargs.get("fulltext_search_max_difference", 2.0)
        self.min_score: float = kwargs.get("fulltext_search_min_score", 1.0)
        self.top_k: int = kwargs.get("fulltext_search_top_k", 10)
        self.max_examples: int = kwargs.get("max_cypher_fewshot_examples", 15)

        
        self._include_types: List[str] = kwargs.get("include_node_types", [])
        self._exclude_types: List[str] = kwargs.get("exclude_node_types", [])
        if len(self._include_types) > 0 and len(self._exclude_types) > 0:
            print("You can provide either of `include_node_types` or `exclude_node_types`.",
                  "By default, `exclude_node_types` has been ignored (set to empty list []).")
            self._exclude_types = []

    @staticmethod
    def generate_full_text_query(input: str) -> str:
        """
        Generate a full-text search query for a given input string.

        This function constructs a query string suitable for a full-text search.
        It processes the input string by splitting it into words and appending a
        similarity threshold (~3 changed characters) to each word, then combines 
        them using the OR operator. Useful for mapping entities from user questions
        to database values, and allows for some misspelings.
        """
        full_text_query: str = ""
        words: List[str] = [el for el in remove_lucene_chars(input).split() if el]
        for word in words[:-1]:
            full_text_query += f" {word}~3 OR"
        full_text_query += f" {words[-1]}~3"
        return full_text_query.strip()

    def retrieve(self, query: str, nresults: int = 100, use_fulltext_search: bool = True) -> str:        
        if use_fulltext_search:
            entities: List[str] = self.entity_chain.invoke({"question": query}).names
            if len(entities) == 0:
                return ""
            # If I can extract entities, then always use fulltext search
            node_ids = []
            for ent in entities:
                output: List[Dict[str, str | float]] = self.graph.query(
                    fulltext_search_cypher, 
                    params={
                        "search_term": self.generate_full_text_query(ent),
                        "min_score": self.min_score,
                        "top_k": self.top_k
                    }
                )
                for i, o in enumerate(output):
                    if i == 0:
                        node_ids.append(o['n'])
                    else:
                        diff: float = o['score'] - output[i-1]['score']
                        if diff >= self.max_difference:
                            break
                        node_ids.append(o['n'])
            rels = self.graph.query(
                f"UNWIND $node_ids AS nid\nMATCH (n: Node {{id: nid}})\n{rels_query}", 
                {"node_ids": node_ids, "limit": nresults}
            )
            docs = self.graph.query(
                f"UNWIND $node_ids AS nid\nMATCH (n: Node {{id: nid}})\n{docs_query}", 
                {"node_ids": node_ids, "limit": nresults}
            )
            rels: str = '\n'.join([rel['output_string'] for rel in rels])
            docs: str = '\n\n'.join([f"{doc['text']}\nSOURCE: {doc['source']}" for doc in docs])
            return f"Nodes Relations: {rels}\nNode Documents:\n{docs}"
        else:
            # TODO: GENERATE CYPHER USING GRAPHCYPHERQACHAIN THAT ALWAYS RETURNS NODE IDS
            examples: List[str] = get_examples(query, max_examples=self.max_examples)
            examples: str = '\n\n'.join(examples)
            graph_schema = construct_schema(
                self.graph.get_structured_schema, self._include_types, self._exclude_types
            )
            cypher: str = self.cypher_generation_chain.invoke(
                {
                    "schema": graph_schema,
                    "examples": examples,
                    "question": query,
                }
            )
            cypher = extract_cypher(cypher)
            if cypher:
                result: List[Dict[str, Any]] = self.graph.query(cypher)[:nresults]
            else:
                result = []
            return '\n'.join(result)
        