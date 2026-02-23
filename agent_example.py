"""
Example usage of the Finance RAG Agent.

This script demonstrates how to use the LangGraph-based agent to answer
financial questions using articles indexed in Elasticsearch.
"""

import os
from dotenv import load_dotenv
from agent.src.rag_agent import FinanceRAGAgent


def main():
    load_dotenv()
    
    # Initialize the agent with local model
    agent = FinanceRAGAgent(
        es_host=os.getenv("ELASTICSEARCH_HOST", "http://localhost:9200"),
        index_name="finance_articles"
        # Uses config.yaml settings by default (google/flan-t5-base)
    )
    
    # Example 1: Single question
    print("=== Example 1: Single Question ===\n")
    result = agent.ask("What are the latest developments with Tesla's Cybertruck?")
    print(f"Q: {result['question']}")
    print(f"A: {result['answer']}\n")
    
    if result['articles_found']:
        print(f"Found {result['num_articles']} relevant articles:")
        for i, article in enumerate(result['articles'][:5], 1):
            print(f"  {i}. {article['title']} ({article['source']})")
    print("\n" + "="*60 + "\n")
    
    # Example 2: Question with no results
    print("=== Example 2: Question Outside Index ===\n")
    result = agent.ask("What is the price of Bitcoin today?")
    print(f"Q: {result['question']}")
    print(f"A: {result['answer']}\n")
    print("="*60 + "\n")
    
    # Example 3: Interactive chat mode
    print("=== Example 3: Interactive Chat ===\n")
    agent.chat()


if __name__ == "__main__":
    main()
