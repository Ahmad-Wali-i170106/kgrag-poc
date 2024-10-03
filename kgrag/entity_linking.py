import os
import re
import json
import requests
from typing import Dict, Any, List, Tuple, Optional
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from kgrag.data_schema_utils import * 

@lru_cache(maxsize=1000)
def wikidata_fetch(url: str, params: frozenset) -> Dict[str, Any]:
    try:
        response = requests.get(url, params=dict(params))
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Error fetching data: {e}")
        return {}

def wikidata_search(query: str) -> List[Dict[str, Any]]:
    if not query:
        return []
    # query = query.replace('(', '').replace(')', '')
    query_strings = set(re.split(r" |_|-|\+|\*|\(|\)|\[|\]", query)) - {''}
    query_strings.add(query)
    if ' ' in query:
        query_strings.add(query.replace(' ',''))


    url = 'https://www.wikidata.org/w/api.php'
    base_params = {
        'action': 'wbsearchentities',
        'format': 'json',
        'language': 'en'
    }

    search_results = set()
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_query = {executor.submit(wikidata_fetch, url, frozenset(base_params.items() | {('search', qs)}))
                           : qs for qs in query_strings}
        for future in as_completed(future_to_query):
            res = future.result()
            if res.get('success'):
                search_results.update({
                    json.dumps({
                        'id': r['id'],
                        'url': r['url'],
                        'label': r.get('label', ''),
                        'aliases': r.get('aliases', []),
                        'description': r.get('description', ''),
                        'type': ''
                    }) for r in res['search']
                })

    if not search_results:
        return []

    search_results = [json.loads(r) for r in search_results]
    search_results = {r['id']: r for r in search_results}

    # Fetch entity types in batches
    batch_size = 50
    all_ids = list(search_results.keys())
    for i in range(0, len(all_ids), batch_size):
        batch_ids = all_ids[i:i+batch_size]
        params = {
            'action': 'wbgetentities',
            'ids': '|'.join(batch_ids),
            'format': 'json',
            'languages': 'en'
        }
        entities = wikidata_fetch(url, frozenset(params.items()))
        if entities.get('success') and 'entities' in entities:
            for id, ent in entities['entities'].items():
                etype_id = next((claim['mainsnak']['datavalue']['value']['id'] 
                                 for claim in ent['claims'].get('P31', [])
                                 if 'mainsnak' in claim and 'datavalue' in claim['mainsnak']), '')
                if etype_id:
                    etype_params = {
                        'action': 'wbgetentities',
                        'ids': etype_id,
                        'format': 'json',
                        'languages': 'en'
                    }
                    etype_data = wikidata_fetch(url, frozenset(etype_params.items()))
                    try:
                        etype = etype_data['entities'][etype_id]['labels']['en']['value']
                    except KeyError:
                        etype = ''
                    search_results[id]['type'] = etype

    return list(search_results.values())

def calculate_cosine_similarity(sentences: List[str], model: SentenceTransformer) -> np.ndarray:
    sentence_embeddings = model.encode(sentences)
    return cosine_similarity([sentence_embeddings[0]], sentence_embeddings[1:]).flatten()

def link_nodes(entities: List[Node], model: Optional[SentenceTransformer] = None, 
               sim_thresh: float = 0.5, verbose: bool = False) -> Tuple[List[Dict[str, Any]], List[Node]]:
    if model is None:
        model = SentenceTransformer(
            'bert-base-nli-mean-tokens',
            cache_folder=os.environ.get("MODELS_CACHE_FOLDER", None),
            tokenizer_kwargs={"clean_up_tokenization_spaces": False}
        )

    matched_nodes, unmatched_nodes = [], []

    for entity in entities:
        res = wikidata_search(entity.id)
        if not res:
            unmatched_nodes.append(entity)
            continue
        if not ' ' in entity.id:
            tcase_id = convert_case(entity.id)
            if tcase_id != entity.id:
                eid = f"{entity.id}/{tcase_id}"
            else:
                eid = entity.id
        else:
            eid = f"{entity.id.replace(' ', '')}/{entity.id}"

        sentences = [f"{eid}, {', '.join(entity.aliases)}, {convert_case(entity.type)}, {entity.definition}"]
        sentences.extend([f"{r['label']}, {', '.join(r['aliases'])}, {r['type']}, {r['description']}".lower() for r in res])
        
        scores = calculate_cosine_similarity(sentences, model)
        best_match_index = np.argmax(scores)

        if scores[best_match_index] < sim_thresh:
            unmatched_nodes.append(entity)
        else:
            best_match = res[best_match_index]
            matched_nodes.append({
                "id": best_match["id"],
                "definition": entity.definition,
                "desc": best_match["description"],
                "type": entity.type,
                "wiki_type": best_match["type"],
                "alias": entity.id,
                'url': best_match['url'].strip('/'),
                "labels": entity.aliases + best_match['aliases'] + [best_match['label']],
                'properties': {p.key: p.value for p in entity.properties}
            })
            if verbose:
                print(f"Matched Node {entity} to Wikidata entity {best_match}\n")

    return matched_nodes, unmatched_nodes

if __name__ == "__main__":
    # q = "OpenAI Generative Pre-trained Transformer"
    # q = "Relation Extraction"
    # q = "BioNLP conference"
    # q = "Pubmed"
    # q = "Pubtator NER"
    q = "Relation Type"
    results = wikidata_search(q)
    for r in results:
        print(r)
        print()
