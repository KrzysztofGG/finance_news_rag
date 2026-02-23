from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import torch


class TextProcessor:
    def __init__(self, ner_model: str = "dslim/bert-base-NER", embedding_model: str = "all-MiniLM-L6-v2"):
        self.device = 0 if torch.cuda.is_available() else -1
        
        self.ner_pipeline = pipeline(
            "ner",
            model=ner_model,
            tokenizer=ner_model,
            aggregation_strategy="simple",
            device=self.device
        )
        
        self.embedding_model = SentenceTransformer(embedding_model)
    
    def extract_entities(self, text: str) -> List[Dict]:
        if not text:
            return []
        
        try:
            entities = self.ner_pipeline(text)
            
            seen = set()
            entity_details = []
            
            for entity in entities:
                entity_type = entity['entity_group']
                entity_word = entity['word']
                
                entity_key = (entity_word, entity_type)
                if entity_key not in seen:
                    seen.add(entity_key)
                    entity_details.append({
                        'text': entity_word,
                        'type': entity_type
                    })
            
            return entity_details
        
        except Exception as e:
            print(f"Error extracting entities: {e}")
            return []
    
    def generate_embedding(self, text: str) -> List[float]:
        if not text:
            return []
        
        try:
            embedding = self.embedding_model.encode(text, convert_to_tensor=False)
            return embedding.tolist()
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return []
    
    def process_article(self, article: Dict) -> Dict:
        full_text = f"{article.get('title', '')} {article.get('description', '')} {article.get('content', '')}"
        
        entities = self.extract_entities(full_text)
        embedding = self.generate_embedding(full_text)
        
        processed_article = article.copy()
        processed_article['entities'] = entities
        processed_article['embedding'] = embedding
        processed_article['full_text'] = full_text
        
        return processed_article
