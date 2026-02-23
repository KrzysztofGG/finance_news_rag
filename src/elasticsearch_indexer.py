from elasticsearch import Elasticsearch, helpers
from typing import List, Dict
import time


class ElasticsearchIndexer:
    def __init__(self, host: str = "http://localhost:9200"):
        self.es = Elasticsearch([host])
        self.wait_for_connection()
    
    def wait_for_connection(self, max_retries: int = 30, retry_delay: int = 2):
        for i in range(max_retries):
            try:
                if self.es.ping():
                    print("Successfully connected to Elasticsearch")
                    return
            except Exception as e:
                print(f"Waiting for Elasticsearch connection... (attempt {i+1}/{max_retries})")
                time.sleep(retry_delay)
        
        raise ConnectionError("Could not connect to Elasticsearch after maximum retries")
    
    def create_index(self, index_name: str):
        if self.es.indices.exists(index=index_name):
            print(f"Index '{index_name}' already exists")
            return
        
        index_mapping = {
            "mappings": {
                "properties": {
                    "title": {"type": "text"},
                    "description": {"type": "text"},
                    "content": {"type": "text"},
                    "full_text": {"type": "text"},
                    "url": {"type": "keyword"},
                    "published_at": {"type": "date"},
                    "source": {"type": "keyword"},
                    "author": {"type": "text"},
                    "company": {"type": "keyword"},
                    "entities": {
                        "type": "nested",
                        "properties": {
                            "text": {"type": "text"},
                            "type": {"type": "keyword"}
                        }
                    },
                    "embedding": {
                        "type": "dense_vector",
                        "dims": 384,
                        "index": True,
                        "similarity": "cosine"
                    }
                }
            }
        }
        
        self.es.indices.create(index=index_name, body=index_mapping)
        print(f"Created index '{index_name}'")
    
    def index_articles(self, articles: List[Dict], index_name: str):
        self.create_index(index_name)
        
        actions = []
        for article in articles:
            action = {
                "_index": index_name,
                "_source": article
            }
            actions.append(action)
        
        try:
            success, failed = helpers.bulk(self.es, actions, raise_on_error=False)
            print(f"Successfully indexed {success} articles")
            if failed:
                print(f"Failed to index {len(failed)} articles")
            return success, failed
        
        except Exception as e:
            print(f"Error during bulk indexing: {e}")
            raise
    
    def search_articles(self, index_name: str, query: str, size: int = 10):
        try:
            response = self.es.search(
                index=index_name,
                body={
                    "query": {
                        "multi_match": {
                            "query": query,
                            "fields": ["title", "description", "content", "full_text"]
                        }
                    },
                    "size": size
                }
            )
            
            return response['hits']['hits']
        
        except Exception as e:
            print(f"Error searching articles: {e}")
            return []
    
    def semantic_search(self, index_name: str, query_text: str, size: int = 10):
        from sentence_transformers import SentenceTransformer
        
        try:
            model = SentenceTransformer('all-MiniLM-L6-v2')
            query_vector = model.encode(query_text).tolist()
            
            response = self.es.search(
                index=index_name,
                body={
                    "knn": {
                        "field": "embedding",
                        "query_vector": query_vector,
                        "k": size,
                        "num_candidates": 100
                    },
                    "_source": ["title", "description", "url", "published_at", "source", "company"]
                }
            )
            
            return response['hits']['hits']
        
        except Exception as e:
            print(f"Error in semantic search: {e}")
            return []
    
    def hybrid_search(self, index_name: str, query: str, size: int = 10, text_weight: float = 0.5):
        from sentence_transformers import SentenceTransformer
        
        try:
            model = SentenceTransformer('all-MiniLM-L6-v2')
            query_vector = model.encode(query).tolist()
            
            response = self.es.search(
                index=index_name,
                body={
                    "query": {
                        "script_score": {
                            "query": {
                                "multi_match": {
                                    "query": query,
                                    "fields": ["title", "description", "content", "full_text"]
                                }
                            },
                            "script": {
                                "source": f"_score * {text_weight} + cosineSimilarity(params.query_vector, 'embedding') * {1 - text_weight}",
                                "params": {"query_vector": query_vector}
                            }
                        }
                    },
                    "size": size
                }
            )
            
            print(f"[DEBUG] Hybrid search for '{query}': found {len(response['hits']['hits'])} results")
            return response
        
        except Exception as e:
            print(f"Error in hybrid search: {e}")
            return {"hits": {"hits": []}}
