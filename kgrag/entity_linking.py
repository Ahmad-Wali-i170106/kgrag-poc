import os
import re
import json
import requests
import itertools
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from kgrag.data_schema_utils import * 
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def wikidata_fetch(params: Dict[str, str]) -> Dict[str, Any] | str:
    url = 'https://www.wikidata.org/w/api.php'
    try:
        data: requests.Response = requests.get(url, params=params)
        if data.status_code == requests.codes.OK:
            return data.json()
        else:
            print(data.status_code, data.content)
            return "There was a problem with getting the results"
    except:
        return 'There was an error'

def wikidata_search(query: str) -> List[Dict[str, Any]]:
    if len(query) == 0:
        return ''
    query_strings: List[str] = re.split(r" |_|-|\+|\*", query)
    # query_strings = [query] + query_strings
    if len(query_strings) > 2:
        st = len(query_strings) - 2
    else:
        st = 0
    query_strings = [' '.join(l) for i in range(st, len(query_strings)) for l in itertools.combinations(query_strings, i+1)]
    search_results = set({})
    for qs in query_strings:
        params = {
            'action': 'wbsearchentities',
            'format': 'json',
            'search': qs,
            'language': 'en'
        }
        res: Dict[str, Any] | str = wikidata_fetch(params)
        if isinstance(res, str):
            continue
        if res['success']:
            search_results.update(set([
                json.dumps(
                    {
                        'id': r['id'],
                        'url': r['url'],
                        'label': r.get('label', ''),
                        'aliases': r.get('aliases', []),
                        'description': r.get('description', ''),
                        'type': ''
                    }
                ) for r in res['search']]
            ))
            # search_results.update(set([(r['id'], r['uri'], r['label'], r['aliases'], '') for r in res['search']]))
    # After getting all the results, get their types
    if len(search_results) == 0:
        return []
    search_results = [json.loads(r) for r in search_results]
    search_results = {r['id']: r for r in search_results}
    ids = '|'.join(search_results.keys())
    params = {
        'action': 'wbgetentities',
        'ids': ids,
        'format': 'json',
        'languages': 'en'
    }
    entities = wikidata_fetch(params)
    if isinstance(entities, dict) and entities.get('success', False) and 'entities' in entities:
        for id, ent in entities['entities'].items():
            # specifically get the entity type of the returned 'entities' from the search results
            etype_id = ent['claims'].get('P31', {})
            if len(etype_id) > 0:
                etype_id = etype_id[0].get('mainsnak', {}).get('datavalue', {}).get('value', {}).get('id', '')
            if len(etype_id) > 0 and id in search_results:
                params['ids'] = etype_id
                d = wikidata_fetch(params)
                try:
                    etype = d.get('entities', {})[etype_id]['labels']['en']['value']
                except:
                    etype = ''
                search_results[id]['type'] = etype
            # search_results[id]['aliases'] = [al.get('value', '') for al in ent['aliases'].get('en', [])]
            
    return list(search_results.values())

def wikipedia_search(query):
    import wikipedia as wd

    try:
        res = wd.search(query)
        if len(res) > 0:
            return res
    except:
        return "There was an error"
    
def calculate_cosine_similarity(sentences: list, model: SentenceTransformer):
    '''
    TODO: Create new similarity function that combines cosine similarity with lexical similarity (levenshstein distance maybe) to improve results

    DOESN'T find match for BERT with 0.7 sim_cutoff
    '''

    # Encoding the sentences to obtain their embeddings
    sentence_embeddings = model.encode(sentences)

    # Calculating the cosine similarity between the first sentence embedding and the rest of the embeddings
    # The result will be a list of similarity scores between the first sentence and each of the other sentences
    similarity_scores = cosine_similarity([sentence_embeddings[0]], sentence_embeddings[1:])
    return similarity_scores.flatten()

def link_nodes(entities: List[Node], model: SentenceTransformer | None = None, sim_thresh: float = 0.5, verbose: bool = False)-> Tuple[List[dict], List[Node]]:
    # Initializing the Sentence Transformer model using BERT with mean-tokens pooling
    if model is None:
        model = SentenceTransformer(
            'bert-base-nli-mean-tokens',
            cache_folder=os.environ.get("MODELS_CACHE_FOLDER", None),
            tokenizer_kwargs={"clean_up_tokenization_spaces": False}
        )
    matched_nodes = []
    unmatched_nodes = []
    for entity in entities:
        res = wikidata_search(entity.id)
        if len(res) == 0:
            unmatched_nodes.append(entity)
            continue
        sentences = [f"{entity.id}, {convert_case(entity.type)}"] #[orig_text]
        sentences.extend([f"{r['label']}, {', '.join(r['aliases'])}, {r['type']}, {r['description']}" for r in res])
        scores = calculate_cosine_similarity(sentences, model)
        ind = np.argmax(scores)
        if scores[ind] < sim_thresh:
            unmatched_nodes.append(entity)
            continue
        matched_nodes.append(
            {
                "id": res[ind]["id"], 
                "desc": res[ind]["description"], 
                "type": entity.type, 
                "wiki_type": res[ind]["type"],
                "alias": entity.id,
                'url': res[ind]['url'],
                "labels": res[ind]['aliases'] + [res[ind]['label']]
            }
        )
        if verbose:
            print(f"Matched Node {entity} to Wikidata entity {res[ind]}")
    return matched_nodes, unmatched_nodes

if __name__ == "__main__":
    import wikipedia as wd
    # q = "OpenAI Generative Pre-trained Transformer"
    q = "Relation Extraction"
    q = "BioNLP conference"
    q = "Pubmed"
    q = "Pubtator NER"
    q = "Relation Type"
    results = wd.search(q)
    if len(results) > 0:
        print(results)
        page = wd.page(results[0], auto_suggest=False)
        print(page.url,'\n',page.summary)
    else:
        results = wikidata_search(q)
        for r in results:
            print(r)
            print()
    # print(wd.search(q))
