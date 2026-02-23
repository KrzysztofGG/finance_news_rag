# Finance RAG Agent

LangGraph-based agent for answering financial questions using Elasticsearch retrieval and Hugging Face LLMs.

## Overview

The Finance RAG Agent uses a stateful workflow to:
1. **Retrieve** relevant articles from Elasticsearch (hybrid search)
2. **Check** if articles were found
3. **Generate** answers using a free Hugging Face LLM or provide helpful fallback

## Quick Start

### Basic Usage

```python
from agent.src.rag_agent import FinanceRAGAgent

# Uses agent/config.yaml by default
agent = FinanceRAGAgent()

# Ask a question
result = agent.ask("What are Tesla's latest developments?")
print(result['answer'])

# Interactive chat
agent.chat()
```

### Run Example Script

```bash
# From project root
uv run agent_example.py
```

## Configuration

The agent uses **YAML configuration** with environment variable overrides.

### Configuration File: `agent/config.yaml`

```yaml
# Elasticsearch Configuration
elasticsearch:
  host: "http://localhost:9200"
  index_name: "finance_articles"

# LLM Configuration
llm:
  model: "mistralai/Mistral-7B-Instruct-v0.3"
  temperature: 0.1
  max_new_tokens: 512

# Retrieval Configuration
retrieval:
  size: 5              # Number of articles to retrieve
  min_score: 0.5       # Minimum relevance score (0.0-1.0)
  text_weight: 0.5     # Hybrid search balance (0.0-1.0)

# Agent Behavior
agent:
  verbose: false
  timeout: 30
```

### Environment Variables

Override config via `.env` (in project root):

```bash
# Required
HUGGINGFACE_API_TOKEN=your_token_here

# Optional overrides
ELASTICSEARCH_HOST=http://localhost:9200
ELASTICSEARCH_INDEX=finance_articles
LLM_MODEL=mistralai/Mistral-7B-Instruct-v0.3
RETRIEVAL_SIZE=5
RETRIEVAL_MIN_SCORE=0.5
RETRIEVAL_TEXT_WEIGHT=0.5
```

**Get Hugging Face token (free):** https://huggingface.co/settings/tokens

### Configuration Priority

1. **Constructor parameters** (highest)
2. **Environment variables**
3. **YAML config file**
4. **Default values** (lowest)

## Customization

### Option 1: Edit config.yaml

```yaml
llm:
  model: "meta-llama/Llama-3.2-3B-Instruct"  # Faster model
  
retrieval:
  size: 10       # More articles
  min_score: 0.3 # Lower threshold
```

### Option 2: Override in Code

```python
agent = FinanceRAGAgent(
    llm_model="microsoft/Phi-3-mini-4k-instruct",
    retrieval_size=10,
    min_score=0.3
)
```

### Option 3: Custom Config File

```python
agent = FinanceRAGAgent(
    config_path="custom_config.yaml"
)
```

### Option 4: Environment Variables

```bash
# In .env
LLM_MODEL=microsoft/Phi-3-mini-4k-instruct
RETRIEVAL_SIZE=10
```

## Recommended LLM Models

All models are **free** via Hugging Face Inference API:

| Model | Speed | Quality | Best For |
|-------|-------|---------|----------|
| `mistralai/Mistral-7B-Instruct-v0.3` | Medium | High | Balanced (default) |
| `meta-llama/Llama-3.2-3B-Instruct` | Fast | Good | Quick responses |
| `microsoft/Phi-3-mini-4k-instruct` | Very Fast | Good | Speed priority |
| `HuggingFaceH4/zephyr-7b-beta` | Medium | High | Instruction following |

## Configuration Parameters

### Retrieval Settings

**`retrieval.size`** (default: 5)
- Number of articles to retrieve from Elasticsearch
- Higher = more context, slower responses
- Recommended: 3-10

**`retrieval.min_score`** (default: 0.5)
- Minimum relevance score threshold (0.0-1.0)
- Higher = stricter relevance, fewer results
- Recommended: 0.3-0.7

**`retrieval.text_weight`** (default: 0.5)
- Hybrid search balance between keyword and semantic
- `0.0` = Pure semantic (meaning-based)
- `0.5` = Balanced (recommended)
- `1.0` = Pure keyword (exact matches)

### LLM Settings

**`llm.model`**
- Hugging Face model ID
- See "Recommended LLM Models" above

**`llm.temperature`** (default: 0.1)
- Creativity/randomness (0.0-1.0)
- Lower = more factual/deterministic
- Recommended: 0.0-0.2 for financial Q&A

**`llm.max_new_tokens`** (default: 512)
- Maximum response length
- Recommended: 256-1024

## Agent Workflow

```
┌─────────────┐
│   Start     │
└──────┬──────┘
       │
       v
┌─────────────────┐
│   Retrieve      │  Hybrid search in Elasticsearch
│   Articles      │  (keyword + semantic)
└──────┬──────────┘
       │
       v
┌─────────────────┐
│  Check Found?   │
└──────┬──────────┘
       │
   ┌───┴───┐
   │       │
  Yes     No
   │       │
   v       v
┌──────┐ ┌──────────────┐
│Answer│ │   Fallback   │
│ LLM  │ │   Message    │
└──────┘ └──────────────┘
```

## API Reference

### `FinanceRAGAgent`

**Constructor:**
```python
agent = FinanceRAGAgent(
    config_path: Optional[str] = None,
    es_host: Optional[str] = None,
    index_name: Optional[str] = None,
    llm_model: Optional[str] = None,
    retrieval_size: Optional[int] = None,
    min_score: Optional[float] = None
)
```

**Methods:**

**`ask(question: str) -> dict`**
```python
result = agent.ask("What are Tesla's latest developments?")

# Returns:
{
    "question": "...",
    "answer": "...",
    "articles_found": True/False,
    "num_articles": 5,
    "articles": [...]  # List of retrieved articles
}
```

**`chat()`**
```python
agent.chat()  # Starts interactive chat loop
```

## Usage Examples

### Example 1: Single Question

```python
from agent.src.rag_agent import FinanceRAGAgent

agent = FinanceRAGAgent()
result = agent.ask("What are Apple's Q4 earnings?")

print(f"Answer: {result['answer']}")
print(f"Based on {result['num_articles']} articles")
```

### Example 2: Custom Configuration

```python
agent = FinanceRAGAgent(
    llm_model="meta-llama/Llama-3.2-3B-Instruct",
    retrieval_size=10,
    min_score=0.3
)

result = agent.ask("What is Microsoft's cloud revenue?")
```

### Example 3: Multiple Questions

```python
agent = FinanceRAGAgent()

questions = [
    "What are Tesla's latest developments?",
    "How is Apple performing in China?",
    "What is Amazon's AWS growth rate?"
]

for q in questions:
    result = agent.ask(q)
    print(f"Q: {q}")
    print(f"A: {result['answer']}\n")
```

### Example 4: Interactive Chat

```python
agent = FinanceRAGAgent()
agent.chat()  # Type questions, 'quit' to exit
```

## Configuration Scenarios

### Fast Responses

```yaml
llm:
  model: "microsoft/Phi-3-mini-4k-instruct"
  max_new_tokens: 256

retrieval:
  size: 3
  min_score: 0.6
```

### High Quality Answers

```yaml
llm:
  model: "mistralai/Mistral-7B-Instruct-v0.3"
  temperature: 0.05
  max_new_tokens: 1024

retrieval:
  size: 10
  min_score: 0.4
```

### Broad Search

```yaml
retrieval:
  size: 15
  min_score: 0.3
  text_weight: 0.3  # More semantic, less keyword
```

### Strict Relevance

```yaml
retrieval:
  size: 5
  min_score: 0.7
  text_weight: 0.7  # More keyword, less semantic
```

## Troubleshooting

### "No articles found"
- Run pipeline to index articles: `uv run pipeline.py --company "Tesla"`
- Lower `min_score` threshold in config
- Try rephrasing your question

### LLM errors
- Check `HUGGINGFACE_API_TOKEN` is set in `.env`
- Verify token is valid: https://huggingface.co/settings/tokens
- Try a different model (some may have rate limits)

### Slow responses
- Use faster model: `microsoft/Phi-3-mini-4k-instruct`
- Reduce `retrieval_size`
- Reduce `max_new_tokens`

### Irrelevant answers
- Increase `min_score` threshold
- Adjust `text_weight` for better search balance
- Index more relevant articles

## Dependencies

The agent requires:
- `langgraph>=0.2.0` - Workflow orchestration
- `langchain-core>=0.3.0` - Base abstractions
- `langchain-huggingface>=0.1.0` - HF integration
- `huggingface-hub>=0.20.0` - API client
- `pyyaml>=6.0` - Config parsing

Plus pipeline dependencies (Elasticsearch, transformers, etc.)

## REST API

You can host the agent as a REST API to invoke it from any frontend application.

### Quick Start

```bash
# Start the API server
./agent/start_api.sh

# Or using uvicorn directly
uvicorn agent.api:app --host 0.0.0.0 --port 8000 --reload
```

API will be available at: **http://localhost:8000**

Interactive docs: **http://localhost:8000/docs**

### Example: Call from JavaScript

```javascript
const response = await fetch('http://localhost:8000/ask', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    question: "What are Tesla's latest developments?"
  })
});

const data = await response.json();
console.log(data.answer);
```

### Example: Call from cURL

```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "What are Tesla'\''s latest developments?"}'
```

**Full API documentation**: See `API.md`

## See Also

- **API Documentation**: `API.md` - REST API usage and examples
- **Main README**: `../README.md` - Pipeline documentation
- **Example Script**: `../agent_example.py` - Usage examples
- **Config File**: `config.yaml` - Default configuration
