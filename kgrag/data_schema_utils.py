import re
import json
from langchain.pydantic_v1 import Field, BaseModel
from typing import List, Optional, Dict, Any

from langchain_core.prompts import PromptTemplate
# from langchain_core.output_parsers import JsonOutputParser

from kgrag.prompts import CHAPTER_EXTRACTION_PROMPT

jsonRegex = r"\{.*\}"

class Property(BaseModel):
    """A single property consisting of key and value"""
    key: str = Field(..., description="key")
    value: str = Field(..., description="value")

class Node(BaseModel):
    id: str = Field(..., description="The identifying property of the node in Title Case")
    type: str = Field(..., description="The entity type / label of the node in PascalCase.")
    properties: Optional[List[Property]] = Field(default=[], description="Detailed properties of the node")
    aliases: List[str] = Field(default=[], description="Alternative names or identifiers for the entity in Title Case")
    definition: Optional[str] = Field(None, description="A concise definition or description of the entity")

class Relationship(BaseModel):
    start_node_id: str = Field(..., description="The id of the first node in the relationship")
    end_node_id: str = Field(..., description="The id of the second node in the relationship")
    type: str = Field(..., description="TThe specific, descriptive label of the relationship in SCREAMING_SNAKE_CASE")
    properties: List[Property] = Field(default=[], description="Detailed properties of the relationship")
    context: Optional[str] = Field(None, description="Additional contextual information about the relationship")
        
class KnowledgeGraph(BaseModel):
    """Generate a knowledge graph with entities and relationships."""
    nodes: list[Node] = Field(
        ..., description="List of nodes in the knowledge graph")
    rels: list[Relationship] = Field(
        ..., description="List of relationships in the knowledge graph"
    )

# Extract entities from text
class Entities(BaseModel):
    """Identifying information about entities."""

    names: List[str] = Field(
        ...,
        description="All the entities that appear in the text",
    )

def _format_property_key(s: str) -> str:
    s = re.sub(r"'|\"|`|\?|%|\$|#|@|\!|\(|\)|\&|\*|\+|\{|\}|\[|\]","", s, flags=re.I)
    words = s.split(' ')
    if len(words) <= 1:
        return s
    first_word = words[0].lower()
    capitalized_words = [word.capitalize() for word in words[1:]]
    return "".join([first_word] + capitalized_words)

def format_nodes(nodes: List[Node]) -> List[Node]:
    return [
        Node(
            id = convert_case(n.id) if isinstance(n.id, str) else n.id,
            type = ''.join([t[0].upper()+t[1:] for t in re.split( r" |_", n.type)]).replace(' ','').replace('_',''),
            properties = [Property(key=_format_property_key(p.key), value=p.value) for p in n.properties if p.key.lower() != 'type'] if n.properties is not None else [],
            definition=n.definition if n.definition is not None else "",
            aliases=[convert_case(a) for a in n.aliases] if n.aliases is not None else []
        )
        for n in nodes
    ]

def format_rels(rels: List[Relationship]) -> List[Relationship]:
    return [
        Relationship(
            start_node_id = convert_case(r.start_node_id) if isinstance(r.start_node_id, str) else r.start_node_id,
            end_node_id = convert_case(r.end_node_id) if isinstance(r.end_node_id, str) else r.start_node_id,
            type = '_'.join(r.type.upper().split(' ')).replace("'", '').replace('`','').replace("%",''),
            properties = [Property(key=_format_property_key(p.key), value=p.value) for p in r.properties if p.key.lower() != 'type'] if r.properties is not None else [],
            context = r.context if r.context is not None else ""
        )
        for r in rels
    ]

def merge_nodes(nodes: List[Node]) -> Node:
    '''
    Merges all the nodes with the same ID and type into a single node

    The only difference in these nodes is their list of properties so 
    I extract and combine a list of unique properties from all these nodes

    Args: 
        nodes (List[Node]):- List of nodes to merge into one
    
    Returns: A single Node object created by merging all the nodes
    '''
    props: List[Property] = []
    definition = ""
    aliases = set([])
    max_def_len = -1
    for node in nodes:
        props.extend([
            p for p in node.properties
            if not p in props
        ])
        if len(node.definition) >= max_def_len:
            # Pick the longest definition (an inaccurate assumption: longest description=most descriptive definition)
            definition = node.definition
            max_def_len = len(node.definition)
        aliases.update(set(node.aliases))
    return Node(id=nodes[0].id, type=nodes[0].type, properties=props, definition=definition, aliases=aliases)


def nodesTextToListOfNodes(nodes_str: List[str]) -> List[Node]:
    nodes: List[Node] = []
    for node in nodes_str:
        nodeList: List[str] = node.split(",")
        if len(nodeList) < 2:
            continue

        name: str = nodeList[0].strip().replace('"', "")
        label: str = nodeList[1].strip().replace('"', "")
        properties: re.Match | None = re.search(jsonRegex, node)
        if properties is None:
            properties = "{}"
        else:
            properties = properties.group(0).replace("True", "true")
        try:
            properties: Dict[str, Any] = json.loads(properties)
            definition: str = properties.pop("definition", "")
            aliases: List[str] = properties.pop("aliases", "").split(',')
            properties = [
                Property(key=k, value=v) for k, v in properties.items()
            ]
            
        except Exception as e:
            print(e)
            properties = []
            definition = ""
            aliases = []
        nodes.append(
            Node(id=name, type=label, properties=properties, definition=definition, aliases=aliases)
        )
    return nodes

def relationshipTextToListOfRelationships(rels_str: List[str]) -> List[Relationship]:
    rels: List[Relationship] = []
    for relation in rels_str:
        relationList: List[str] = relation.split(",")
        if len(relation) < 3:
            continue
        start: str = relationList[0].strip().replace('"', "")
        end: str = relationList[2].strip().replace('"', "")
        type: str = relationList[1].strip().replace('"', "")

        properties: re.Match[str] | None = re.search(jsonRegex, relation)
        if properties is None:
            properties = "{}"
        else:
            properties = properties.group(0).replace("True", "true")
        try:
            properties = json.loads(properties)
            context = properties.pop("context", "")
            properties = [
                Property(key=k, value=v) for k, v in properties.items()
            ]
        except:
            context = ""
            properties = []
        rels.append(
            Relationship(start_node_id=start, end_node_id=end, type=type, properties=properties, context=context)
        )
    return rels

##############################
# Util for parsing documents #
##############################


class PageMetadata(BaseModel):
    chapter_title: str = Field(description="The extracted chapter or section title, or an empty string if none found")



def extract_chapter(text: str, model):
    # parser = JsonOutputParser(pydantic_object=PageMetadata)
    # template = chapter_extraction_prompt #"Please find if the input text from the user is the first page from a new chapter. If so, extract chapter title from the input text. Otherwise, return an empty string.\nHint: Look for a heading in either of the first 3 lines of the text.\n{format_instructions}\n{text}"
    # user_message = f"""{text}"""
    # format_instructions = parser.get_format_instructions()
    prompt = PromptTemplate.from_template(template=CHAPTER_EXTRACTION_PROMPT) #, partial_variables={"format_instructions":format_instructions}

    chain = prompt | model
    
    res = chain.invoke({"text": text}).content
    try:
        res = json.loads(res)
    except:
        print(res)
        res = re.findall(": *(.+)}", res, flags=re.I)[-1].strip('\n ').strip('"').strip('\n ')
        res = {"chapter_title": res}
    return res

def is_number(n: str) -> bool:
    return all([ni in "1234567890" for ni in n])

def camel_case_to_normal(name: str) -> str:
    name = re.sub('(.)([A-Z][a-z]+)', r'\1 \2', name)
    name = re.sub('  ([A-Z])', r' \1', name)
    name = re.sub('([a-z0-9])([A-Z])', r'\1 \2', name)
    return name.lower()

def convert_case(text: str) -> str:
    # Add space before any uppercase letter that follows a lowercase letter
    # or a number, and before any number that follows a letter
    if is_number(text):
        return text
    pattern = re.compile(r'(?<!^)(?=[A-Z][a-z]|\d)')
    text = pattern.sub(' ', text)
    
    # Add space before any uppercase letter that follows another uppercase letter
    # and is followed by a lowercase letter
    pattern = re.compile(r'([A-Z])([A-Z][a-z])')
    text = pattern.sub(r'\1 \2', text)
    
    # Add space before any number that follows a letter
    pattern = re.compile(r'([a-zA-Z])(\d)')
    text = pattern.sub(r'\1 \2', text)
    text = re.sub(r" {2,}", " ", text)
    return text