import os
import json
from typing import List, Dict


class JSONHandler:
    @staticmethod
    def _resolve_path(file_name: str, for_write: bool = False) -> str:
        """Resolve file path using JSON_DIR env var; create directory if needed when writing."""
        base_dir = os.getenv("JSON_DIR")

        # If already absolute, don't prefix
        if os.path.isabs(file_name):
            resolved = file_name
        else:
            resolved = os.path.join(base_dir, file_name) if base_dir else file_name

        if for_write:
            os.makedirs(os.path.dirname(resolved) or ".", exist_ok=True)

        return resolved

    @staticmethod
    def save_to_json(articles: List[Dict], output_file: str):
        try:
            resolved_path = JSONHandler._resolve_path(output_file, for_write=True)
            with open(resolved_path, 'a', encoding='utf-8') as f:
                for article in articles:
                    json.dump(article, f, ensure_ascii=False)
                    f.write('\n')
            
            print(f"Successfully appended {len(articles)} articles to {resolved_path}")
        
        except Exception as e:
            print(f"Error saving to JSON: {e}")
            raise
    
    @staticmethod
    def load_from_json(input_file: str) -> List[Dict]:
        articles = []
        try:
            resolved_path = JSONHandler._resolve_path(input_file, for_write=False)
            with open(resolved_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        articles.append(json.loads(line))
            
            print(f"Successfully loaded {len(articles)} articles from {resolved_path}")
            return articles
        
        except Exception as e:
            print(f"Error loading from JSON: {e}")
            return []
