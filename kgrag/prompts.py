
# - **Suggested Relationship Types**: Following are some suggested relationship types/labels that you can use: {rel_types}
DATA_EXTRACTION_SYSTEM = """# Knowledge Graph Instructions
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


CYPHER_GENERATION_SYSTEM = """Task: Generate Cypher statement to query a graph database.
Instructions:
Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.
Schema:
{schema}
Note: Do not include any explanations or apologies in your responses.
Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.
Do not include any text except the generated Cypher statement.
{examples}
"""

OCR_SYSTEM = """You are an OCR engine that is very good at extracting legible text from scanned PDF pages.
You must be at least 90% confident that the extracted text is accurate. Ignore the text otherwise. Be especially careful with digits or numerical values and units."""