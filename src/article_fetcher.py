import os
from newsapi import NewsApiClient
from typing import List, Dict
from datetime import datetime, timedelta


class ArticleFetcher:
    def __init__(self, api_key: str):
        self.newsapi = NewsApiClient(api_key=api_key)
    
    def fetch_articles(self, company_name: str, days_back: int = 1, language: str = 'en', limit: int = 50) -> List[Dict]:
        from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        to_date = datetime.now().strftime('%Y-%m-%d')
        
        try:
            all_articles = self.newsapi.get_everything(
                q=company_name,
                from_param=from_date,
                to=to_date,
                language=language,
                sort_by='relevancy',
                page_size=100
            )
            
            articles = []
            for article in all_articles.get('articles', []):
                if limit and len(articles) >= limit:
                    break
                if article.get('content') or article.get('description'):
                    articles.append({
                        'title': article.get('title', ''),
                        'description': article.get('description', ''),
                        'content': article.get('content', ''),
                        'url': article.get('url', ''),
                        'published_at': article.get('publishedAt', ''),
                        'source': article.get('source', {}).get('name', ''),
                        'author': article.get('author', ''),
                        'company': company_name
                    })
            
            return articles
        
        except Exception as e:
            print(f"Error fetching articles: {e}")
            return []
