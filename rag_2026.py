"""
2026 Gold Standard RAG System
==============================
Production-grade RAG with observability, cost tracking, and resilience.
"""

import time

from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    KnowledgeGraphIndex,
    Settings,
    load_index_from_storage,
    Document
)
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.llms.openai import OpenAI
from llama_index.llms.anthropic import Anthropic
from llama_index.llms.google_genai import GoogleGenerativeAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import LLMRerank

# Import our 2026 gold standard modules
from config import get_config
from observability import initialize_observability, get_logger, get_metrics
from cost_tracker import get_tracker


class GoldStandardRAG:
    """2026 Gold Standard RAG System."""
    
    def __init__(self, incremental: bool = True):
        """Initialize the RAG system."""
        self.config = get_config()
        self.logger = get_logger()
        self.metrics = get_metrics()
        self.cost_tracker = get_tracker()
        
        # Initialize observability
        self.logger.logger.info("Initializing 2026 Gold Standard RAG System")
        
        # Configure LlamaIndex settings
        self._configure_settings()
        
        # Build or load indices
        self.vector_index, self.kg_index, self.nodes = self._build_indices(incremental)
        
        # Create query engine
        self.query_engine = self._create_query_engine()
        
        self.logger.logger.info("RAG System ready", extra={
            "event": "system_ready",
            "num_nodes": len(self.nodes)
        })
    
    def _configure_settings(self):
        """Configure global LlamaIndex settings."""
        # Embedding model
        if self.config.embedding.provider == "openai":
            Settings.embed_model = OpenAIEmbedding(
                model=self.config.embedding.model,
                max_retries=self.config.embedding.max_retries
            )
        elif self.config.embedding.provider == "huggingface":
            Settings.embed_model = HuggingFaceEmbedding(
                model_name=self.config.embedding.model
            )
        else:
            raise ValueError(f"Unsupported embedding provider: {self.config.embedding.provider}")

        # Primary LLM with automatic fallback chain
        llm_configured = False

        # Try primary provider first
        if self.config.llm.primary_provider == "anthropic" and self.config.anthropic_api_key:
            try:
                Settings.llm = Anthropic(
                    model=self.config.llm.primary_model,
                    max_retries=self.config.llm.max_retries,
                    temperature=self.config.llm.temperature,
                    timeout=self.config.llm.timeout
                )
                self.logger.logger.info(f"Using primary: Anthropic {self.config.llm.primary_model}")
                llm_configured = True
            except Exception as e:
                self.logger.logger.warning(f"Anthropic failed: {e}")

        elif self.config.llm.primary_provider == "openai" and self.config.openai_api_key:
            try:
                Settings.llm = OpenAI(
                    model=self.config.llm.primary_model,
                    max_retries=self.config.llm.max_retries,
                    temperature=self.config.llm.temperature,
                    timeout=self.config.llm.timeout
                )
                self.logger.logger.info(f"Using primary: OpenAI {self.config.llm.primary_model}")
                llm_configured = True
            except Exception as e:
                self.logger.logger.warning(f"OpenAI failed: {e}")

        elif self.config.llm.primary_provider == "gemini" and self.config.google_api_key:
            try:
                Settings.llm = GoogleGenerativeAI(
                    model=self.config.llm.primary_model,
                    max_retries=self.config.llm.max_retries,
                    temperature=self.config.llm.temperature
                )
                self.logger.logger.info(f"Using primary: Gemini {self.config.llm.primary_model}")
                llm_configured = True
            except Exception as e:
                self.logger.logger.warning(f"Gemini failed: {e}")

        # Try fallback providers
        if not llm_configured:
            for provider in self.config.llm.fallback_providers:
                model = self.config.llm.fallback_models.get(provider)

                if provider == "openrouter" and self.config.openrouter_api_key:
                    try:
                        from llama_index.llms.openrouter import OpenRouter as OpenRouterLLM
                        Settings.llm = OpenRouterLLM(
                            model=model,
                            api_key=self.config.openrouter_api_key,
                            temperature=0
                        )
                        self.logger.logger.warning(f"Using fallback: OpenRouter {model} (FREE)")
                        llm_configured = True
                        break
                    except Exception as e:
                        self.logger.logger.warning(f"OpenRouter fallback failed: {e}")

                elif provider == "gemini" and self.config.google_api_key:
                    try:
                        Settings.llm = GoogleGenerativeAI(model=model, temperature=0)
                        self.logger.logger.warning(f"Using fallback: Gemini {model} (FREE)")
                        llm_configured = True
                        break
                    except Exception as e:
                        self.logger.logger.warning(f"Gemini fallback failed: {e}")

                elif provider == "openai" and self.config.openai_api_key:
                    try:
                        Settings.llm = OpenAI(model=model, temperature=0)
                        self.logger.logger.warning(f"Using fallback: OpenAI {model}")
                        llm_configured = True
                        break
                    except Exception as e:
                        self.logger.logger.warning(f"OpenAI fallback failed: {e}")

                elif provider == "anthropic" and self.config.anthropic_api_key:
                    try:
                        Settings.llm = Anthropic(model=model, temperature=0)
                        self.logger.logger.warning(f"Using fallback: Anthropic {model}")
                        llm_configured = True
                        break
                    except Exception as e:
                        self.logger.logger.warning(f"Anthropic fallback failed: {e}")

        if not llm_configured:
            raise RuntimeError("No LLM provider available! Check API keys and providers.")
        
        self.logger.logger.info("Settings configured", extra={
            "event": "settings_configured",
            "embedding_model": self.config.embedding.model,
            "llm_model": self.config.llm.primary_model
        })
    
    def _build_indices(self, incremental: bool):
        """Build or load vector and knowledge graph indices."""
        storage_dir = self.config.storage.storage_dir
        docs_dir = self.config.storage.docs_dir
        
        # Load documents
        self.logger.logger.info(f"Loading documents from {docs_dir}")
        reader = SimpleDirectoryReader(
            input_dir=str(docs_dir),
            required_exts=[".md"],
            recursive=True,
            filename_as_id=True
        )
        documents = reader.load_data()
        
        # Sanitize content
        cleaned_docs = []
        for doc in documents:
            clean_text = "".join(filter(lambda x: x.isprintable() or x in "\n\r\t", doc.text))
            new_doc = Document(text=clean_text, metadata=doc.metadata)
            cleaned_docs.append(new_doc)
        
        documents = cleaned_docs
        
        # Parse into nodes
        parser = MarkdownNodeParser()
        nodes = parser.get_nodes_from_documents(documents)
        total_nodes = len(nodes)
        
        self.logger.logger.info(f"Parsed {total_nodes} nodes")
        
        # Track embedding costs
        total_chars = sum(len(node.get_content()) for node in nodes)
        estimated_tokens = total_chars // 4  # rough estimate
        self.cost_tracker.track_embedding(estimated_tokens, "openai", self.config.embedding.model)
        
        # Build/load vector index
        if not storage_dir.exists():
            storage_dir.mkdir(parents=True, exist_ok=True)
            self.logger.logger.info("Building vector index...")
            vector_index = VectorStoreIndex(nodes)
            vector_index.set_index_id("vector")
            vector_index.storage_context.persist(persist_dir=str(storage_dir))
            self.metrics.record_query("index_build", 0)
        else:
            self.logger.logger.info("Loading existing vector index...")
            storage_context = StorageContext.from_defaults(persist_dir=str(storage_dir))
            vector_index = load_index_from_storage(storage_context, index_id="vector")
        
        # Build/load knowledge graph index
        kg_provider = self.config.llm.kg_provider
        if kg_provider == "openai":
            kg_llm = OpenAI(
                model=self.config.llm.kg_model,
                max_retries=self.config.llm.max_retries,
                temperature=0
            )
        elif kg_provider == "anthropic":
            kg_llm = Anthropic(
                model=self.config.llm.kg_model,
                max_retries=self.config.llm.max_retries,
                temperature=0
            )
        elif kg_provider == "gemini":
            kg_llm = GoogleGenerativeAI(
                model=self.config.llm.kg_model,
                max_retries=self.config.llm.max_retries,
                temperature=0
            )
        else:
            raise ValueError(f"Unsupported KG LLM provider: {kg_provider}")
        
        try:
            storage_context = StorageContext.from_defaults(persist_dir=str(storage_dir))
            kg_index = load_index_from_storage(storage_context, index_id="kg")
            self.logger.logger.info("Loaded existing knowledge graph index")
            
            if incremental:
                kg_index.refresh_ref(documents)
                kg_index.storage_context.persist(persist_dir=str(storage_dir))
        except Exception:
            self.logger.logger.info("Building knowledge graph index (this may take a while)...")
            kg_index = KnowledgeGraphIndex(
                [],
                storage_context=storage_context,
                llm=kg_llm,
                max_triplets_per_chunk=self.config.retriever.kg_max_triplets_per_chunk,
                include_embeddings=self.config.retriever.kg_include_embeddings
            )
            kg_index.set_index_id("kg")
            
            # Process in batches with checkpoints
            checkpoint_interval = self.config.storage.checkpoint_interval
            for i in range(0, total_nodes, checkpoint_interval):
                batch = nodes[i : i + checkpoint_interval]
                self.logger.logger.info(f"Processing KG batch {i}/{total_nodes}...")
                
                kg_index.insert_nodes(batch)
                kg_index.storage_context.persist(persist_dir=str(storage_dir))
                
                # Track KG extraction costs
                batch_tokens = sum(len(node.get_content()) // 4 for node in batch)
                self.cost_tracker.track_llm_call(
                    batch_tokens,
                    batch_tokens // 2,  # estimate output tokens
                    self.config.llm.kg_provider,
                    self.config.llm.kg_model
                )
        
        return vector_index, kg_index, nodes
    
    def _create_query_engine(self):
        """Create the query engine with fusion retrieval and reranking."""
        # Create retrievers
        vector_retriever = self.vector_index.as_retriever(
            similarity_top_k=self.config.retriever.vector_top_k
        )
        kg_retriever = self.kg_index.as_retriever(
            similarity_top_k=self.config.retriever.kg_top_k
        )
        bm25_retriever = BM25Retriever.from_defaults(
            nodes=self.nodes,
            similarity_top_k=self.config.retriever.bm25_top_k
        )
        
        # Fusion retriever
        fusion_retriever = QueryFusionRetriever(
            [vector_retriever, kg_retriever, bm25_retriever],
            num_queries=self.config.retriever.fusion_num_queries,
            mode=self.config.retriever.fusion_mode,
            use_async=True
        )
        
        # Reranker
        rerank_provider = self.config.llm.rerank_provider
        if rerank_provider == "openai":
            rerank_llm = OpenAI(
                model=self.config.llm.rerank_model,
                max_retries=self.config.llm.max_retries,
                temperature=0
            )
        elif rerank_provider == "anthropic":
            rerank_llm = Anthropic(
                model=self.config.llm.rerank_model,
                max_retries=self.config.llm.max_retries,
                temperature=0
            )
        else:
            raise ValueError(
                f"Unsupported rerank_provider '{rerank_provider}'. "
                "Supported providers are: 'openai', 'anthropic'."
            )
        reranker = LLMRerank(
            choice_batch_size=self.config.retriever.rerank_batch_size,
            top_n=self.config.retriever.rerank_top_n,
            llm=rerank_llm
        )
        
        # Query engine
        query_engine = RetrieverQueryEngine.from_args(
            fusion_retriever,
            node_postprocessors=[reranker]
        )
        
        return query_engine
    
    def query(self, query_text: str):
        """Execute a query with full observability."""
        start_time = time.time()
        
        self.logger.log_query(query_text)
        
        try:
            # Execute query
            response = self.query_engine.query(query_text)
            
            # Calculate metrics
            latency = time.time() - start_time
            num_sources = len(response.source_nodes)
            
            # Log and record metrics
            self.logger.log_retrieval(
                num_sources,
                [node.node.metadata.get("file_name", "unknown") for node in response.source_nodes],
                latency * 1000
            )
            
            self.metrics.record_query("success", latency)
            self.metrics.record_retrieval(num_sources)
            
            # Estimate and track costs
            # (In production, you'd get actual token counts from the API response)
            estimated_input_tokens = len(query_text) // 4 + sum(len(node.node.get_content()) // 4 for node in response.source_nodes[:5])
            estimated_output_tokens = len(str(response)) // 4
            
            self.cost_tracker.track_llm_call(
                estimated_input_tokens,
                estimated_output_tokens,
                self.config.llm.primary_provider,
                self.config.llm.primary_model
            )
            
            self.metrics.record_llm_tokens(
                self.config.llm.primary_provider,
                estimated_input_tokens,
                estimated_output_tokens
            )
            
            # Update cost gauge
            daily_cost = self.cost_tracker.get_daily_cost()
            self.metrics.update_daily_cost(daily_cost)
            
            return response
            
        except Exception as e:
            latency = time.time() - start_time
            self.logger.log_error(e, {"query": query_text})
            self.metrics.record_query("error", latency)
            raise


def build_gold_standard_rag(incremental: bool = True) -> GoldStandardRAG:
    """Build the 2026 Gold Standard RAG system."""
    # Initialize observability first
    initialize_observability()
    
    # Create and return RAG system
    return GoldStandardRAG(incremental=incremental)


if __name__ == "__main__":
    import sys
    
    # Build the system
    print("Building 2026 Gold Standard RAG System...")
    rag = build_gold_standard_rag(incremental=True)
    
    # Interactive query mode
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        response = rag.query(query)
        print("\n" + "="*60)
        print("Response:")
        print("="*60)
        print(response)
        print("\n" + "="*60)
        print("Cost Report:")
        print("="*60)
        print(rag.cost_tracker.get_report())
    else:
        print("\nInteractive mode. Type 'quit' to exit.")
        while True:
            query = input("\nQuery: ")
            if query.lower() in ["quit", "exit", "q"]:
                break
            
            response = rag.query(query)
            print("\n" + "="*60)
            print(response)
