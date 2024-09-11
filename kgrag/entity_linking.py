import re
import json
import requests
import itertools
from typing import Dict, Any, List, Tuple, Optional
import wikipedia as wd


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
        st = len(query_strings) - 3
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
                        'aliases': r.get('aliases', ''),
                        'description': r.get('description', ''),
                        'type': ''
                    }
                ) for r in res['search']]
            ))
            # search_results.update(set([(r['id'], r['uri'], r['label'], r['aliases'], '') for r in res['search']]))
    # After getting all the results, get their types
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
    try:
        res = wd.search(query)
        if len(res) > 0:
            return res
    except:
        return "There was an error"

if __name__ == "__main__":
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
