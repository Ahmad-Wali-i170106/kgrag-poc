
# - **Suggested Relationship Types**: Following are some suggested relationship types/labels that you can use: {rel_types}
DATA_EXTRACTION_SYSTEM2 = """# Knowledge Graph Instructions
## 1. Overview
You are a top-tier algorithm designed for extracting information in structured formats to build a knowledge graph.
You must extract information to build a knowledge graph that will serve as a knowledge base for an educational assistant.
The resulting knowledge graph will be used to create syllabus & course contents, summarise chapters, explain theorems etc. 
You must extract the information from consecutive samples of text from a much larger text, most likely a book or short tutorial.
You must extract information that seems most relevant to the overall subject theme of the text.
{subject}
- **Nodes** represent entities and concepts. They're akin to Wikipedia/Wikidata nodes. They are assigned a type which is akin to entity type for entities. In general, proper nouns should be used for creating nodes.
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
- A **relationship** is between two nodes which are present in the list of nodes.
- Only create and return a relationship between two nodes if you are more than 95% certain that the relationship exists.
- **Relationship Type Naming Convention**: Relationship types should be written in SCREAMING_SNAKE_CASE.
## 4. Handling Numerical Data and Dates
- Numerical data, like age or other related information, should be incorporated as attributes or properties of the respective nodes.
- **No Separate Nodes for Dates/Numbers**: Do not create separate nodes for dates or numerical values. Always attach them as attributes or properties of nodes.
- **Property Format**: Properties must be in a key-value format.
- **Quotation Marks**: Never use escaped single or double quotes within property values.
- **Naming Convention**: Use camelCase for property keys.
## 5. Coreference Resolution
- **Maintain Entity Consistency**: When extracting entities, it's vital to ensure consistency.
- If an entity, such as "John Doe", is mentioned multiple times in the text but is referred to by different names or pronouns (e.g., "Joe", "he"), 
always use the most complete identifier for that entity throughout the knowledge graph. In this example, use "John Doe" as the entity ID.  
- If an entity, such as 'Artificial Intelligence' is mentioned multiple times in the text but is referred to by different names, pronouns or acronyms (e.g., "AI", "A.I"),
always use the most complete identifier for that entity throughout the knowledge graph. In this example, use "Artificial Intelligence" as the entity ID.
Remember, the knowledge graph should be coherent and easily understandable, so maintaining consistency in entity references is crucial. 
## 6. Strict Compliance
Adhere to the rules strictly. Non-compliance will result in termination."""

# - **Type Property**: NEVER include the `type` property in the list of properties.


DATA_EXTRACTION_SYSTEM = """# Knowledge Graph Extraction for Rich Information Retrieval
## 1. Overview
You are an advanced algorithm designed to extract knowledge from various types of informational content. Your task is to build a knowledge graph that will be linked to Wikidata and serve as a comprehensive knowledge base for a question-answering system.
{subject}
## 2. Content Focus
- Extract detailed information about concepts, entities, processes, and their relationships from the given text.
- Prioritize information that provides rich context and is likely to be useful for answering a wide range of questions.
- Include relevant attributes, properties, and descriptive information for each extracted entity.

## 3. Node Extraction
- **Node IDs**: Use clear, unambiguous identifiers in Title Case. Avoid integers, abbreviations, and acronyms.
- **Node Types**: Use PascalCase. Be as specific and descriptive as possible to aid in Wikidata matching.
- Include all relevant attributes of the entity in the node properties.
- Extract and include alternative names or aliases for entities when present in the text.
- Following are some existing node/entity types that were extracted from previous samples of the same document:\n{node_types}

## 4. Relationship Extraction
- Use SCREAMING_SNAKE_CASE for relationship types.
- Create detailed, informative relationship types that clearly describe the nature of the connection.
- Only create relationships between two existing nodes with high certainty (>90% confidence) 
- Include directional relationships where applicable (e.g., PRECEDED_BY, FOLLOWED_BY instead of just RELATED_TO).

## 5. Contextual Information
- For each node and relationship, strive to capture contextual information that might be useful for answering questions.
- Include temporal information when available (e.g., dates, time periods, sequence of events).
- Capture geographical or spatial information if relevant.

## 6. Handling Definitions and Descriptions
- For key concepts, include concise definitions or descriptions as node properties.
- Capture any notable characteristics, functions, or use cases of entities.

## 7. Coreference and Consistency
- Maintain consistent entity references throughout the graph.
- Resolve coreferences to their most complete form, including potential aliases or alternative names.

## 8. Granularity
- Strike a balance between detailed extraction and maintaining a coherent graph structure.
- Create separate nodes for distinct concepts, even if closely related, to allow for precise linking and querying.

Remember, the goal is to create a knowledge graph that provides rich, contextual information to support accurate entity linking with Wikidata and to provide comprehensive information for the QA agent to generate informed answers."""


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
You must be at least 90% confident that the extracted text is accurate. Ignore the text otherwise. Be especially careful with digits or numerical values and units.
Return markdown-formatted text as output. Remember, accuracy is crucial. It's better to express uncertainty or return nothing than to make incorrect assumptions."""

CHAPTER_EXTRACTION_PROMPT = """You are an expert text analyzer specializing in book structure and formatting. Your task is to determine if the given text represents the start of a new chapter, section, or front matter (like a foreword or preface) and extract the title if present.

Instructions:
1. Carefully examine the first 5 lines of the provided text.
2. Look for patterns indicating a new chapter, section, or front matter, such as:
   - Markdown headings of level 1 or 2 (e.g., "## Foreword" or "# Chapter 1")
   - Lines containing only a number or "Chapter" followed by a number
   - Prominent, centered text that appears to be a title
   - All-caps or bold text that could be a title
3. If you identify a title, extract it and remove any markdown formatting, numbers, or words like "Chapter".
4. For front matter (like Foreword, Preface, Introduction), include these words in the extracted title.
5. If no title is found, return an empty string.

Rules:
- The extracted title can be up to 10 words to accommodate front matter titles.
- Ignore page numbers, headers, footers, or signatures at the end of the text.
- Don't include subtitles or epigraphs as part of the title.
- If the title is in markdown format, remove the markdown symbols (e.g., "#", "**") but keep the text.

Output your result in the following JSON format:
{{
    "chapter_title": "extracted title or empty string"
}}

Text to analyze:
{text}
"""