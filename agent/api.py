"""
FastAPI REST API for the Finance RAG Agent.

This provides HTTP endpoints to interact with the agent from external applications.
"""

import os
import sys
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from agent.src.rag_agent import FinanceRAGAgent

load_dotenv()

app = FastAPI(
    title="Finance RAG Agent API",
    description="REST API for answering financial questions using RAG",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

agent = None


class QuestionRequest(BaseModel):
    question: str = Field(..., description="The financial question to ask")
    retrieval_size: Optional[int] = Field(None, description="Number of articles to retrieve")
    min_score: Optional[float] = Field(None, description="Minimum relevance score")


class QuestionResponse(BaseModel):
    question: str
    answer: str
    articles_found: bool
    num_articles: int
    articles: list


class HealthResponse(BaseModel):
    status: str
    elasticsearch_connected: bool
    agent_initialized: bool


@app.on_event("startup")
async def startup_event():
    """Initialize the agent on startup."""
    global agent
    try:
        agent = FinanceRAGAgent()
        print("✓ Agent initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize agent: {e}")
        raise


@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Finance RAG Agent API",
        "version": "1.0.0",
        "endpoints": {
            "POST /ask": "Ask a financial question",
            "GET /health": "Check API health status",
            "GET /config": "Get current configuration",
            "GET /docs": "Interactive API documentation"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check the health status of the API and its dependencies."""
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    es_connected = False
    try:
        es_connected = agent.indexer.es.ping()
    except Exception:
        pass
    
    return HealthResponse(
        status="healthy" if es_connected else "degraded",
        elasticsearch_connected=es_connected,
        agent_initialized=agent is not None
    )


@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """
    Ask a financial question and get an answer based on indexed articles.
    
    The agent will:
    1. Retrieve relevant articles from Elasticsearch
    2. Generate an answer using the LLM
    3. Return the answer with source articles
    """
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    try:
        # Create temporary agent with custom params if provided
        if request.retrieval_size or request.min_score:
            temp_agent = FinanceRAGAgent(
                retrieval_size=request.retrieval_size,
                min_score=request.min_score
            )
            result = temp_agent.ask(request.question)
        else:
            result = agent.ask(request.question)
        
        return QuestionResponse(**result)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")


@app.get("/config", response_model=dict)
async def get_config():
    """Get current agent configuration."""
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    return {
        "elasticsearch": {
            "index": agent.index_name
        },
        "retrieval": {
            "size": agent.retrieval_size,
            "min_score": agent.min_score,
            "text_weight": agent.text_weight
        },
        "verbose": agent.verbose
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
