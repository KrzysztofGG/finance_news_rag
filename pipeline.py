import os
import argparse
from dotenv import load_dotenv
from src.article_fetcher import ArticleFetcher
from src.text_processor import TextProcessor
from src.json_handler import JSONHandler
from src.elasticsearch_indexer import ElasticsearchIndexer


def main():
    parser = argparse.ArgumentParser(description='Finance Article Processing Pipeline')
    parser.add_argument('--company', type=str, required=True, help='Company name to search for')
    parser.add_argument('--days', type=int, default=7, help='Number of days to look back')
    parser.add_argument('--output', type=str, default='articles.jsonl', help='Output JSON file')
    parser.add_argument('--skip-fetch', action='store_true', help='Skip fetching and use existing JSON file')
    parser.add_argument('--skip-index', action='store_true', help='Skip Elasticsearch indexing')
    parser.add_argument('--index-name', type=str, default='finance_articles', help='Elasticsearch index name')
    
    args = parser.parse_args()
    
    load_dotenv()
    
    if not args.skip_fetch:
        print(f"\n=== Step 1: Fetching articles for '{args.company}' ===")
        api_key = os.getenv('NEWS_API_KEY')
        if not api_key:
            raise ValueError("NEWS_API_KEY not found in environment variables. Please set it in .env file")
        
        fetcher = ArticleFetcher(api_key)
        articles = fetcher.fetch_articles(args.company, days_back=args.days)
        print(f"Fetched {len(articles)} articles")
        
        if not articles:
            print("No articles found. Exiting.")
            return
        
        print(f"\n=== Step 2: Processing articles (NER + Embeddings) ===")
        processor = TextProcessor()
        processed_articles = []
        
        for i, article in enumerate(articles, 1):
            print(f"Processing article {i}/{len(articles)}: {article['title'][:50]}...")
            processed_article = processor.process_article(article)
            processed_articles.append(processed_article)
        
        print(f"\n=== Step 3: Saving to JSON file ===")
        JSONHandler.save_to_json(processed_articles, args.output)
    else:
        print(f"\n=== Loading articles from {args.output} ===")
        processed_articles = JSONHandler.load_from_json(args.output)
        
        if not processed_articles:
            print("No articles found in JSON file. Exiting.")
            return
    
    if not args.skip_index:
        print(f"\n=== Step 4: Indexing to Elasticsearch ===")
        es_host = os.getenv('ELASTICSEARCH_HOST', 'http://localhost:9200')
        
        indexer = ElasticsearchIndexer(host=es_host)
        success, failed = indexer.index_articles(processed_articles, args.index_name)
        
        print(f"\n=== Pipeline Complete ===")
        print(f"Total articles processed: {len(processed_articles)}")
        print(f"Successfully indexed: {success}")
        print(f"Failed to index: {len(failed) if failed else 0}")
        print(f"Elasticsearch index: {args.index_name}")
        print(f"Kibana dashboard: http://localhost:5601")
    else:
        print(f"\n=== Pipeline Complete (Indexing Skipped) ===")
        print(f"Total articles processed: {len(processed_articles)}")
        print(f"Output file: {args.output}")


if __name__ == "__main__":
    main()
