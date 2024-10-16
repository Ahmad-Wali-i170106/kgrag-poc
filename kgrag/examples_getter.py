import os
from typing import List, Dict, Any, Optional
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings

class ExamplesGetter:
    
    def __init__(
            self, 
            # use_milvus: bool = False, 
            json_filename: str | None = None, 
            sim_model: Optional[SentenceTransformer | Embeddings] = None,
            milvus_kwargs: dict = {}, 
            **kwargs
        ):
        
        use_milvus = False
        self.sim_model = sim_model
        if self.sim_model is None:
            self.sim_model = SentenceTransformer(
                kwargs.get("embed_model_name", 'sentence-transformers/all-MiniLM-L6-v2'), 
                cache_folder=os.environ.get("MODELS_CACHE_FOLDER", None),
                tokenizer_kwargs={"clean_up_tokenization_spaces": False}
            )


        # Read from the JSON filename to get the list of examples
        self.examples = None
        if json_filename is not None:
            import json
            
            with open(json_filename, "r") as f:
                self.examples: List[Dict[str, str]] = json.load(f)
        
        if use_milvus:
            # TODO: Init Milvus Database API
            raise NotImplementedError("Haven't yet implemented `ExamplesGetter` to use Milvus")
        else:
            if self.examples is None:
                raise ValueError("You must provide the `json_filename` when `use_milvus` is set to False")
            # self.examples = examples
            self._embeddings = self.sim_model.encode([q['question'] for q in self.examples])
    
    @staticmethod
    def format_example(example: Dict[str, str]) -> str:
        return f"""# QUESTION: {example['question']}
CYPHER: ```{example['cypher']}```
"""
    
    def _embed(self, sents: List[str]) -> List[List[float]]:
        if isinstance(self.sim_model, SentenceTransformer):
            return [list(sent) for sent in self.sim_model.encode(sents)]
        elif isinstance(self.sim_model, Embeddings):
            return self.sim_model.embed_documents(sents)

    def get_examples(self, query: str, top_k: int = 20, sim_cutoff: float = 0.5) -> List[str]:
        if self.examples is not None:
            similarities = cosine_similarity(self._embed([query]), self._embeddings).flatten()
            examples = [(self.examples[idx], sim) for idx, sim in enumerate(similarities) if sim >= sim_cutoff]
            examples = sorted(examples, key=lambda x: x[1], reverse=True)
            return [self.format_example(ex) for ex, score in examples][:top_k]
        else:
            # TODO: Query Milvus and return `top_k` similar examples
            return []
        
    def update_from_file(self, json_filename: str, keep_old=False) -> None:
        if self.examples is None:
            raise Exception("This method is only meant for the case when examples are loaded from a JSON file - Not stored in Milvus")
        import json
        with open(json_filename, "r") as f:
            examples: List[Dict[str, str]] = json.load(f)
        new_embeddings = self._embed([e['question'] for e in examples])
        if keep_old:
            self.examples += examples
            self._embeddings += new_embeddings
        else:
            self.examples = examples
            self._embeddings = new_embeddings
    
    def add_examples(self, examples: List[Dict[str, str]]) -> None:
        if self.examples is not None:
            if len(examples) == 0:
                return
            new_embeddings = self._embed([e['question'] for e in examples])
            self.examples += examples
            self._embeddings += new_embeddings
        else:
            # TODO: Add new examples to Milvus
            return
        
        
        