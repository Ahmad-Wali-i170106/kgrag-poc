'''
Contains code to remove duplicate nodes and relationships if they have similar node IDs
Uses an LLM with JSON output and input to remove duplicates.
'''

import json
import re
from itertools import groupby

from kgrag.data_schema_utils import (
    Node,
    Relationship,
    Property,
    nodesTextToListOfNodes,
    relationshipTextToListOfRelationships,
)

from typing import List, Dict, Any, Tuple

from langchain_core.messages import AnyMessage
from langchain_core.language_models import BaseLanguageModel

try:
    from langchain_google_genai.llms import _BaseGoogleGenerativeAI
except ImportError:
    _BaseGoogleGenerativeAI = None

def generate_system_message_for_nodes() -> str:
    return """Your task is to identify if there are duplicated nodes and if so merge them into one nod. Only merge the nodes that refer to the same entity.
You will be given different datasets of nodes and some of these nodes may be duplicated or refer to the same entity. 
Some of these nodes may be referred by an acronym or abbreviation. You should expand that acronym according to what matches best with the type.
The datasets contains nodes in the form [ENTITY_ID, TYPE, PROPERTIES]. When you have completed your task please give me the 
resulting nodes in the same format. Only return the nodes and relationships no other text. If there is no duplicated nodes return the original nodes.

Here is an example of the input you will be given:
["alice", "Person", {"age": 25, "occupation": "lawyer", "name":"Alice"}], ["bob", "Person", {"occupation": "journalist", "name": "Bob"}], ["alice.com", "Webpage", {"url": "www.alice.com"}], ["bob.com", "Webpage", {"url": "www.bob.com"}]
"""


def generate_system_message_for_relationships() -> str:
    return """
Your task is to identify if a set of relationships make sense.
If they do not make sense please remove them from the dataset.
Some relationships may be duplicated or refer to the same entity. 
Please merge relationships that refer to the same entity.
The datasets contains relationships in the form [ENTITY_ID_1, RELATIONSHIP, ENTITY_ID_2, PROPERTIES].
You will also be given a set of ENTITY_IDs that are valid.
Some relationships may use ENTITY_IDs that are not in the valid set but refer to a entity in the valid set.
If a relationships refer to a ENTITY_ID in the valid set please change the ID so it matches the valid ID.
When you have completed your task please give me the valid relationships in the same format. Only return the relationships no other text.

Here is an example of the input you will be given:
["alice", "roommate", "bob", {"start": 2021}], ["alice", "owns", "alice.com", {}], ["bob", "owns", "bob.com", {}]
"""

def generate_prompt(data) -> str:
    return f""" Here is the data:
{data}
"""

internalRegex = r"\[(.*?)\]"


class DataDisambiguation:
    def __init__(self, llm: BaseLanguageModel, group_by_label: bool = True) -> None:
        self.llm = llm
        self.group_by_label = group_by_label

    def run(self, nodes: List[Node], relationships: List[Relationship]) -> Tuple[List[Node], List[Relationship]]:
        
        nodes: List[Dict[str, Any]] = [
            {
                'name': node.id,
                'label': node.type,
                'properties': {p.key: p.value for p in node.properties} | {"definition": node.definition, "aliases": ','.join(node.aliases)}
            }
            for node in nodes
        ]
        relationships: List[Dict[str, str | dict]] = [
            {
                "start": rel.start_node_id,
                "end": rel.end_node_id,
                "type": rel.type,
                "properties": {p.key: p.value for p in rel.properties} | {"context": rel.context}
            }
            for  rel in relationships
        ]

        nodes = sorted(nodes, key=lambda x: x["label"])
        
        new_nodes: list[Node] = []
        new_relationships: List[Relationship] = []
        if self.group_by_label:
            node_groups = groupby(nodes, lambda x: x["label"])
        else:
            # If not group_by_label, just place all the nodes in one large group
            node_groups = [["Node", nodes]]
        for group in node_groups:
            disString: str = ""
            nodes_in_group = list(group[1])
            if len(nodes_in_group) == 1:
                props: dict = nodes_in_group[0]['properties']
                definition = props.pop('definition', '')
                aliases = props.pop('aliases', '').split(',')
                new_nodes.append(
                    Node(
                        id=nodes_in_group[0]['name'], 
                        type=nodes_in_group[0]['label'],
                        properties=[Property(key=k, value=v) for k, v in props.items()],
                        definition=definition,
                        aliases=aliases
                    )
                )
                continue

            for node in nodes_in_group:
                disString += (
                    '["'
                    + node["name"]
                    + '", "'
                    + node["label"]
                    + '", '
                    + json.dumps(node["properties"])
                    + "]\n"
                )

            if _BaseGoogleGenerativeAI is not None and isinstance(self.llm, _BaseGoogleGenerativeAI) and 'gemini-1.0' in self.llm.model:
                role = "human"
            else:
                role = "system"

            messages: list[dict[str, str]] = [
                {"role": role, "content": generate_system_message_for_nodes()},
                {"role": "human", "content": generate_prompt(disString)},
            ]
            rawNodes = self.llm.invoke(messages)
            if isinstance(rawNodes, AnyMessage):
                rawNodes = rawNodes.content
            elif not isinstance(rawNodes, str):
                raise TypeError(f"Got unexpected type: {type(rawNodes)} from llm output; Expected `AIMessage` or `str`")

            n: list[str] = re.findall(internalRegex, rawNodes)

            new_nodes.extend(nodesTextToListOfNodes(n))
        
        relationship_data = "Relationships:\n"
        for relation in relationships:
            relationship_data += (
                '["'
                + relation["start"]
                + '", "'
                + relation["type"]
                + '", "'
                + relation["end"]
                + '", '
                + json.dumps(relation["properties"])
                + "]\n"
            )

        node_labels: List[str] = [node.id for node in new_nodes]
        relationship_data += "Valid Nodes:\n" + "\n".join(node_labels)

        messages = [
            {
                "role": role,
                "content": generate_system_message_for_relationships(),
            },
            {"role": "human", "content": generate_prompt(relationship_data)},
        ]
        rawRelationships: str = self.llm.invoke(messages)
        if isinstance(rawRelationships, AnyMessage):
            rawRelationships = rawRelationships.content
        elif not isinstance(rawRelationships, str):
            raise TypeError(f"Got unexpected type: {type(rawRelationships)} from llm output; Expected `AIMessage` or `str`")
        rels: List[str] = re.findall(internalRegex, rawRelationships)
        new_relationships.extend(relationshipTextToListOfRelationships(rels))
        
        return new_nodes, new_relationships
