"""Configuration loader for the Finance RAG Agent."""

import os
import yaml
from typing import Dict, Any
from pathlib import Path


class AgentConfig:
    """Configuration manager for the RAG agent."""
    
    DEFAULT_CONFIG = {
        "elasticsearch": {
            "host": "http://localhost:9200",
            "index_name": "finance_articles"
        },
        "llm": {
            "model": "mistralai/Mistral-7B-Instruct-v0.3",
            "temperature": 0.1,
            "max_new_tokens": 512
        },
        "retrieval": {
            "size": 5,
            "min_score": 0.5,
            "text_weight": 0.5
        },
        "agent": {
            "verbose": False,
            "timeout": 30
        }
    }
    
    def __init__(self, config_path: str = None):
        """
        Initialize configuration.
        
        Args:
            config_path: Path to YAML config file. If None, uses default config.yaml
                        in the agent directory.
        """
        self.config = self.DEFAULT_CONFIG.copy()
        
        if config_path is None:
            # Default to config.yaml in agent directory
            agent_dir = Path(__file__).parent.parent
            config_path = agent_dir / "config.yaml"
        
        if os.path.exists(config_path):
            self._load_from_file(config_path)
        
        # Override with environment variables if set
        self._load_from_env()
    
    def _load_from_file(self, config_path: str):
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                file_config = yaml.safe_load(f)
                if file_config:
                    self._merge_config(self.config, file_config)
        except Exception as e:
            print(f"Warning: Could not load config from {config_path}: {e}")
    
    def _load_from_env(self):
        """Override config with environment variables."""
        # Elasticsearch
        if os.getenv("ELASTICSEARCH_HOST"):
            self.config["elasticsearch"]["host"] = os.getenv("ELASTICSEARCH_HOST")
        if os.getenv("ELASTICSEARCH_INDEX"):
            self.config["elasticsearch"]["index_name"] = os.getenv("ELASTICSEARCH_INDEX")
        
        # LLM
        if os.getenv("LLM_MODEL"):
            self.config["llm"]["model"] = os.getenv("LLM_MODEL")
        if os.getenv("LLM_TEMPERATURE"):
            self.config["llm"]["temperature"] = float(os.getenv("LLM_TEMPERATURE"))
        if os.getenv("LLM_MAX_TOKENS"):
            self.config["llm"]["max_new_tokens"] = int(os.getenv("LLM_MAX_TOKENS"))
        
        # Retrieval
        if os.getenv("RETRIEVAL_SIZE"):
            self.config["retrieval"]["size"] = int(os.getenv("RETRIEVAL_SIZE"))
        if os.getenv("RETRIEVAL_MIN_SCORE"):
            self.config["retrieval"]["min_score"] = float(os.getenv("RETRIEVAL_MIN_SCORE"))
        if os.getenv("RETRIEVAL_TEXT_WEIGHT"):
            self.config["retrieval"]["text_weight"] = float(os.getenv("RETRIEVAL_TEXT_WEIGHT"))
    
    def _merge_config(self, base: Dict, override: Dict):
        """Recursively merge override config into base config."""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_config(base[key], value)
            else:
                base[key] = value
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path (e.g., "llm.model")
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key_path.split(".")
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    @property
    def es_host(self) -> str:
        """Get Elasticsearch host."""
        return self.get("elasticsearch.host")
    
    @property
    def es_index(self) -> str:
        """Get Elasticsearch index name."""
        return self.get("elasticsearch.index_name")
    
    @property
    def llm_model(self) -> str:
        """Get LLM model name."""
        return self.get("llm.model")
    
    @property
    def llm_temperature(self) -> float:
        """Get LLM temperature."""
        return self.get("llm.temperature")
    
    @property
    def llm_max_tokens(self) -> int:
        """Get LLM max tokens."""
        return self.get("llm.max_new_tokens")
    
    @property
    def retrieval_size(self) -> int:
        """Get retrieval size."""
        return self.get("retrieval.size")
    
    @property
    def retrieval_min_score(self) -> float:
        """Get minimum retrieval score."""
        return self.get("retrieval.min_score")
    
    @property
    def retrieval_text_weight(self) -> float:
        """Get retrieval text weight for hybrid search."""
        return self.get("retrieval.text_weight")
    
    @property
    def verbose(self) -> bool:
        """Get verbose flag."""
        return self.get("agent.verbose", False)
    
    @property
    def timeout(self) -> int:
        """Get timeout in seconds."""
        return self.get("agent.timeout", 30)
    
    def __repr__(self) -> str:
        """String representation of config."""
        return f"AgentConfig({self.config})"
