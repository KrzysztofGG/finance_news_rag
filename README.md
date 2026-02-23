# Finance Article RAG Pipeline

A complete pipeline for fetching financial news articles, processing them with NER (Named Entity Recognition), generating embeddings, and indexing them in Elasticsearch. Includes a LangGraph-based RAG agent for answering questions using the indexed articles.

## Features

### Pipeline
- **Article Fetching**: Downloads finance articles from NewsAPI (free tier)
- **NER Processing**: Extracts named entities using HuggingFace transformers
- **Text Embeddings**: Generates 384-dimensional vector embeddings
- **JSON Storage**: Saves processed articles in JSONL format
- **Elasticsearch Integration**: Hybrid search (keyword + semantic)
- **Kibana Dashboard**: Visualize and explore your data

### RAG Agent
- **LangGraph Workflow**: Stateful agent with retrieval and generation nodes
- **Free LLM**: Uses Hugging Face Inference API (no OpenAI key needed)
- **Configuration**: YAML-based config with environment variable overrides
- **Interactive Chat**: Built-in chat interface for Q&A

See `agent/README.md` for agent documentation.

## Project Structure

```
finance_rag/
├── src/                         # Pipeline modules
│   ├── article_fetcher.py       # NewsAPI integration
│   ├── text_processor.py        # NER and embedding generation
│   ├── json_handler.py          # JSON read/write operations
│   └── elasticsearch_indexer.py # Elasticsearch operations
├── agent/                       # RAG agent (separate module)
│   ├── src/
│   │   ├── rag_agent.py         # LangGraph agent implementation
│   │   └── config_loader.py     # Configuration management
│   ├── config.yaml              # Agent configuration
│   └── README.md                # Agent documentation
├── pipeline.py                  # Main pipeline script
├── agent_example.py             # Agent usage examples
├── docker-compose.yml           # Elasticsearch + Kibana setup
├── pyproject.toml               # Dependencies
└── README.md                    # This file
```

## Setup

### 1. Install Dependencies

**Option A: Using uv (Recommended - Fast!)**

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync dependencies (creates venv automatically)
uv sync

# Or install dependencies without creating a project
uv pip install -r requirements.txt
```

**Option B: Using pip**

```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Create a `.env` file and make sure it contains:
```bash
# Required for pipeline
NEWS_API_KEY=your_newsapi_key_here

# Required for agent
HUGGINGFACE_API_TOKEN=your_huggingface_token_here

# Optional (defaults shown)
ELASTICSEARCH_HOST=http://localhost:9200
ELASTICSEARCH_INDEX=finance_articles
JSON_DIR=data
```

**Get API keys:**
- NewsAPI (free): https://newsapi.org/register
- Hugging Face (optional, for embeddings): https://huggingface.co/settings/tokens

### 3. Install and Setup Ollama (for RAG Agent)

The agent uses Ollama for local LLM inference. Install and set it up:

**Install Ollama:**
```bash
# Linux
curl -fsSL https://ollama.com/install.sh | sh

# macOS
brew install ollama

# Windows: Download from https://ollama.com/download
```

**Start Ollama:**
```bash
ollama serve
```

**Pull the model (gemma2:2b recommended):**
```bash
ollama pull gemma2:2b
```

**Other model options:**
- `gemma2:2b` - Very fast, 1.6GB (default)
- `llama3.2:3b` - Fast, 2GB
- `phi3:mini` - Fast, 2.3GB
- `mistral:7b` - Best quality, 4.1GB

**Manage models:**
```bash
# List installed models
ollama list

# Remove a model
ollama rm gemma2:2b

# Clean up HuggingFace cache (if you tried transformers before)
rm -rf ~/.cache/huggingface/hub/
```

### 4. Start Elasticsearch & Kibana

```bash
docker-compose up -d
```

Wait ~30 seconds for Elasticsearch to be ready. Check status:
```bash
curl http://localhost:9200
```

## Usage

### Basic Usage

Fetch articles for a company and index them:

**With uv:**
```bash
uv run pipeline.py --company "Tesla"
```

**With python:**
```bash
python pipeline.py --company "Tesla"
```

### Advanced Options

**With uv:**
```bash
uv run pipeline.py \
  --company "Apple" \
  --days 14 \
  --output apple_articles.jsonl \
  --index-name apple_finance
```

**With python:**
```bash
python pipeline.py \
  --company "Apple" \
  --days 14 \
  --output apple_articles.jsonl \
  --index-name apple_finance
```

### Command-Line Arguments

- `--company` (required): Company name to search for
- `--days`: Number of days to look back (default: 7)
- `--output`: Output JSON file path (default: articles.jsonl)
- `--skip-fetch`: Skip fetching and use existing JSON file
- `--skip-index`: Skip Elasticsearch indexing
- `--index-name`: Elasticsearch index name (default: finance_articles)

### Example Workflows

**1. Fetch and process only (no indexing):**
```bash
python pipeline.py --company "Microsoft" --skip-index
```

**2. Index existing JSON file:**
```bash
python pipeline.py --company "Microsoft" --skip-fetch --output existing_articles.jsonl
```

**3. Process multiple companies:**
```bash
python pipeline.py --company "Google" --index-name google_articles
python pipeline.py --company "Amazon" --index-name amazon_articles
```

## Output Format

Each line in the JSONL file contains:

```json
{
  "title": "Article title",
  "description": "Article description",
  "content": "Article content",
  "url": "https://...",
  "published_at": "2024-01-01T00:00:00Z",
  "source": "Source name",
  "author": "Author name",
  "company": "Company name",
  "full_text": "Combined text",
  "entities": [
    {"text": "Tesla", "type": "ORG"},
    {"text": "Elon Musk", "type": "PER"}
  ],
  "embedding": [0.123, -0.456, ...]
}
```

## Elasticsearch Index Mapping

The pipeline automatically creates an index with:

- **Text fields**: title, description, content, full_text
- **Keyword fields**: url, source, company
- **Nested entities**: text, type
- **Dense vector**: 384-dimensional embeddings with cosine similarity

## Accessing Your Data

### Kibana Dashboard
Open http://localhost:5601 in your browser to:
- Visualize article trends
- Search and filter articles
- Create custom dashboards

### Elasticsearch API

**1. Keyword/Text Search (multi_match):**
```bash
curl -X GET "localhost:9200/finance_articles/_search?pretty" -H 'Content-Type: application/json' -d'
{
  "query": {
    "match": {
      "full_text": "earnings report"
    }
  }
}
'
```

**2. Semantic Search (kNN vector similarity):**
```python
from src.elasticsearch_indexer import ElasticsearchIndexer

indexer = ElasticsearchIndexer()
results = indexer.semantic_search(
    index_name="finance_articles",
    query_text="What are the latest financial results?",
    size=10
)

for hit in results:
    print(f"Score: {hit['_score']:.4f}")
    print(f"Title: {hit['_source']['title']}")
    print(f"URL: {hit['_source']['url']}\n")
```

**3. Hybrid Search (combines text + semantic):**
```python
from src.elasticsearch_indexer import ElasticsearchIndexer

indexer = ElasticsearchIndexer()
results = indexer.hybrid_search(
    index_name="finance_articles",
    query="revenue growth",
    size=10,
    text_weight=0.5  # 50% text match, 50% semantic similarity
)

for hit in results:
    print(f"Score: {hit['_score']:.4f}")
    print(f"Title: {hit['_source']['title']}\n")
```

### Search Methods Comparison

| Method | Use Case | How It Works |
|--------|----------|-------------|
| **search_articles** | Exact keyword matching | Uses BM25 algorithm to find documents with matching words |
| **semantic_search** | Meaning-based search | Uses vector embeddings to find semantically similar content |
| **hybrid_search** | Best of both worlds | Combines keyword matching + semantic similarity with adjustable weights |

**Example:** Query "company profits"
- **Keyword search**: Finds articles containing "company" AND "profits"
- **Semantic search**: Also finds articles about "revenue", "earnings", "financial performance" (similar meaning)
- **Hybrid search**: Balances both approaches for optimal results

## Models Used

- **NER**: `dslim/bert-base-NER` - Recognizes PER (person), ORG (organization), LOC (location), MISC
- **Embeddings**: `all-MiniLM-L6-v2` - 384-dimensional sentence embeddings

## Troubleshooting

### Elasticsearch connection failed
```bash
# Check if Elasticsearch is running
docker ps

# View logs
docker logs finance_elasticsearch

# Restart services
docker-compose restart
```

### NewsAPI rate limit
Free tier allows 100 requests/day. Reduce `--days` or use `--skip-fetch` to reprocess existing data.

### Out of memory
Reduce batch size or use CPU instead of GPU by setting:
```python
export CUDA_VISIBLE_DEVICES=""
```

## Cleanup

Stop and remove containers:
```bash
docker-compose down
```

Remove data volumes:
```bash
docker-compose down -v
```

## RAG Agent Usage

### Python API

After indexing articles, use the agent to answer questions:

```python
from agent.src.rag_agent import FinanceRAGAgent

# Initialize with config file (agent/config.yaml)
agent = FinanceRAGAgent()

# Ask a question
result = agent.ask("What are Tesla's latest developments?")
print(result['answer'])

# Or start interactive chat
agent.chat()
```

**Run the example:**
```bash
uv run agent_example.py
```

### REST API (for Frontend Applications)

Host the agent as a REST API to call from any frontend:

```bash
# Start the API server
./agent/start_api.sh

# API available at: http://localhost:8000
# Interactive docs: http://localhost:8000/docs
```

**Call from JavaScript:**
```javascript
const response = await fetch('http://localhost:8000/ask', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ question: "What are Tesla's latest developments?" })
});
const data = await response.json();
console.log(data.answer);
```

**Full documentation:**
- Agent usage: `agent/README.md`
- REST API: `agent/API.md`

## License

MIT
