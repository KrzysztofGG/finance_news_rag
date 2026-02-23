import os
import sys
from typing import TypedDict, Annotated, Sequence, Optional
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
import requests

# Add parent directory to path to import from pipeline src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.elasticsearch_indexer import ElasticsearchIndexer
from agent.src.config_loader import AgentConfig


class AgentState(TypedDict):
    """State for the RAG agent."""
    messages: Annotated[Sequence[BaseMessage], "The conversation messages"]
    question: str
    retrieved_articles: list
    answer: str
    articles_found: bool


class FinanceRAGAgent:
    """LangGraph agent for answering financial questions using Elasticsearch retrieval."""
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        es_host: Optional[str] = None,
        index_name: Optional[str] = None,
        llm_model: Optional[str] = None,
        retrieval_size: Optional[int] = None,
        min_score: Optional[float] = None
    ):
        # Load configuration from file
        self.config = AgentConfig(config_path)
        
        # Override with explicit parameters if provided
        es_host = es_host or self.config.es_host
        self.index_name = index_name or self.config.es_index
        llm_model = llm_model or self.config.llm_model
        self.retrieval_size = retrieval_size or self.config.retrieval_size
        self.min_score = min_score or self.config.retrieval_min_score
        
        self.indexer = ElasticsearchIndexer(host=es_host)
        
        # Use Ollama for local model inference
        self.llm_model = llm_model
        self.llm_temperature = self.config.llm_temperature
        self.llm_max_tokens = self.config.llm_max_tokens
        self.ollama_url = "http://localhost:11434/api/generate"
        
        # Check if Ollama is running
        try:
            response = requests.get("http://localhost:11434/api/tags")
            if response.status_code == 200:
                print(f"✓ Ollama is running, using model: {llm_model}")
            else:
                print(f"⚠ Ollama may not be running properly")
        except Exception as e:
            print(f"⚠ Warning: Could not connect to Ollama at http://localhost:11434")
            print(f"  Make sure Ollama is installed and running: 'ollama serve'")
        
        self.text_weight = self.config.retrieval_text_weight
        self.verbose = self.config.verbose
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(AgentState)
        
        workflow.add_node("retrieve", self._retrieve_articles)
        workflow.add_node("generate_answer", self._generate_answer)
        workflow.add_node("no_articles_found", self._no_articles_found)
        
        workflow.set_entry_point("retrieve")
        
        workflow.add_conditional_edges(
            "retrieve",
            self._check_articles_found,
            {
                "found": "generate_answer",
                "not_found": "no_articles_found"
            }
        )
        
        workflow.add_edge("generate_answer", END)
        workflow.add_edge("no_articles_found", END)
        
        return workflow.compile()
    
    def _retrieve_articles(self, state: AgentState) -> AgentState:
        """Retrieve relevant articles from Elasticsearch using hybrid search."""
        question = state["question"]
        
        try:
            results = self.indexer.hybrid_search(
                index_name=self.index_name,
                query=question,
                size=self.retrieval_size,
                text_weight=self.text_weight
            )

            articles = []
            for hit in results.get("hits", {}).get("hits", []):
                score = hit["_score"]
                if score >= self.min_score:
                    source = hit["_source"]
                    articles.append({
                        "title": source.get("title", ""),
                        "description": source.get("description", ""),
                        "content": source.get("content", ""),
                        "url": source.get("url", ""),
                        "source": source.get("source", ""),
                        "published_at": source.get("published_at", ""),
                        "score": score
                    })
            
            state["retrieved_articles"] = articles
            state["articles_found"] = len(articles) > 0
            
        except Exception as e:
            if self.verbose:
                print(f"Error retrieving articles: {e}")
            state["retrieved_articles"] = []
            state["articles_found"] = False
        
        return state
    
    def _check_articles_found(self, state: AgentState) -> str:
        """Check if any relevant articles were found."""
        return "found" if state["articles_found"] else "not_found"
    
    def _generate_answer(self, state: AgentState) -> AgentState:
        """Generate answer using LLM based on retrieved articles."""
        question = state["question"]
        articles = state["retrieved_articles"]
        
        context = self._format_context(articles)
        
        prompt = f"""You are a financial analyst assistant. Answer the user's question based ONLY on the provided article excerpts.

Question: {question}

Relevant Articles:
{context}

Instructions:
- Provide a clear, concise answer based on the articles above
- Cite specific articles when making claims (e.g., "According to [Source Name]...")
- If the articles don't fully answer the question, acknowledge what information is available
- Be factual and avoid speculation

Answer:"""
        
        try:
            # Use Ollama for local model inference
            response = requests.post(
                self.ollama_url,
                json={
                    "model": self.llm_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": self.llm_temperature,
                        "num_predict": self.llm_max_tokens
                    }
                },
                timeout=60
            )
            
            if response.status_code == 200:
                answer = response.json().get('response', '')
            else:
                answer = f"Error: Ollama returned status {response.status_code}"
            
            state["answer"] = answer
            state["messages"].append(HumanMessage(content=question))
            state["messages"].append(AIMessage(content=answer))
            
        except Exception as e:
            error_msg = f"Error generating answer: {e}"
            if self.verbose:
                print(error_msg)
            state["answer"] = error_msg
        
        return state
    
    def _no_articles_found(self, state: AgentState) -> AgentState:
        """Handle case when no relevant articles are found."""
        question = state["question"]
        
        answer = f"""I couldn't find any relevant articles in the database to answer your question: "{question}"

This could mean:
- No articles matching your query have been indexed yet
- The question topic is outside the scope of the indexed financial articles
- Try rephrasing your question or asking about a different company/topic

You can run the pipeline to fetch and index more articles:
```
uv run pipeline.py --company "YourCompany"
```"""
        
        state["answer"] = answer
        state["messages"].append(HumanMessage(content=question))
        state["messages"].append(AIMessage(content=answer))
        
        return state
    
    def _format_context(self, articles: list) -> str:
        """Format retrieved articles into context string."""
        context_parts = []
        
        for i, article in enumerate(articles, 1):
            context_parts.append(f"""
Article {i} (Score: {article['score']:.2f}):
Source: {article['source']}
Title: {article['title']}
Published: {article['published_at']}
Content: {article['description']} {article['content'][:500]}...
URL: {article['url']}
""")
        
        return "\n".join(context_parts)
    
    def ask(self, question: str) -> dict:
        """Ask a question and get an answer based on indexed articles."""
        initial_state = {
            "messages": [],
            "question": question,
            "retrieved_articles": [],
            "answer": "",
            "articles_found": False
        }
        
        final_state = self.graph.invoke(initial_state)
        
        return {
            "question": question,
            "answer": final_state["answer"],
            "articles_found": final_state["articles_found"],
            "num_articles": len(final_state["retrieved_articles"]),
            "articles": final_state["retrieved_articles"]
        }
    
    def chat(self):
        """Interactive chat loop."""
        print("Finance RAG Agent - Ask questions about indexed financial articles")
        print("Type 'quit' or 'exit' to stop\n")
        
        while True:
            question = input("You: ").strip()
            
            if question.lower() in ["quit", "exit"]:
                print("Goodbye!")
                break
            
            if not question:
                continue
            
            print("\nThinking...\n")
            result = self.ask(question)
            
            print(f"Agent: {result['answer']}\n")
            
            if result["articles_found"]:
                print(f"(Based on {result['num_articles']} articles)\n")
