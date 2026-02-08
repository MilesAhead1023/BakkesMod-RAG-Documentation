# BakkesMod RAG - GUI User Guide

## 🚀 Quick Start

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/MilesAhead1023/BakkesMod-RAG-Documentation.git
   cd BakkesMod-RAG-Documentation
   ```

2. **Set up environment:**
   ```bash
   # Copy environment template
   cp .env.example .env
   
   # Edit .env and add your API keys
   # Required: OPENAI_API_KEY, ANTHROPIC_API_KEY
   ```

3. **Launch the GUI:**
   
   **Linux/Mac:**
   ```bash
   ./start_gui.sh
   ```
   
   **Windows:**
   ```cmd
   start_gui.bat
   ```
   
   The GUI will automatically open in your browser at `http://localhost:7860`

## 📚 Using the GUI

### Tab 1: Query Documentation

Ask questions about BakkesMod SDK and get accurate, source-cited answers.

**Features:**
- **Real-time streaming**: See answers as they're generated
- **Semantic caching**: Instant responses for similar questions (saves costs)
- **Confidence scores**: Know how reliable the answer is
- **Source citations**: See which documentation files were used

**Example Questions:**
- "What is BakkesMod and how does it work?"
- "How do I hook the goal scored event?"
- "How do I access the player's car velocity vector?"
- "What ImGui functions are available for creating UI?"
- "How do I register console commands?"

**Tips:**
- Enable "Use semantic cache" for faster responses on repeated queries
- Check confidence scores - VERY HIGH (85%+) is most reliable
- Review source files to verify the answer
- Be specific in your questions for better results

### Tab 2: Generate Plugin Code

Create complete, validated BakkesMod plugins from natural language descriptions.

**Workflow:**
1. **Describe your plugin** in the requirements box
2. **Click "Generate Code"** to create header and implementation files
3. **Review the code** in both panels (syntax-highlighted)
4. **Check validation status** - warnings will appear if detected
5. **Enter a plugin name** and click "Export to Files"
6. **Find your files** in `generated_plugins/YourPluginName/`

**Example Requirements:**

*Simple Plugin:*
```
Create a plugin that logs a message when a goal is scored,
including the scorer's name and team color.
```

*Advanced Plugin:*
```
Create a plugin with ImGui window that displays:
- Current boost amount
- Ball velocity
- Player rotation
Hook tick event to update every frame.
Add console command to toggle the window.
```

**Generated Files:**
- `PluginName.h` - Header with class declaration
- `PluginName.cpp` - Full implementation
- `README.md` - Documentation and build instructions

### Tab 3: Statistics

Monitor your usage and system performance.

**Metrics:**
- **Total queries**: Number of questions asked
- **Success rate**: Percentage of successful queries
- **Average query time**: Typical response time
- **Cache hits**: Questions answered from cache
- **Cache hit rate**: Percentage of cached responses
- **Cost savings**: Estimated savings from caching

**Refresh** the statistics anytime to see updated metrics.

### Tab 4: Help

View this documentation and system information within the GUI.

## 🎯 Features

### Hybrid Retrieval System

The RAG system uses three complementary retrieval methods:

1. **Vector Search**: Semantic similarity using embeddings
2. **BM25 Keyword Search**: Traditional keyword matching
3. **Knowledge Graph**: Relationship-based traversal

Results are fused using reciprocal rank fusion for optimal relevance.

### Semantic Caching

- **Cache similar queries**: 92% similarity threshold
- **Cost savings**: 30-40% reduction in API costs
- **Instant responses**: ~50ms for cache hits
- **7-day TTL**: Entries expire after one week

### Code Generation

- **RAG-Enhanced**: Uses documentation for accurate API calls
- **Template-Based**: Follows BakkesMod conventions
- **Validated**: Automatic syntax and API validation
- **Complete Projects**: Generates full plugin structure

### Quality Assurance

**Confidence Scoring:**
- **VERY HIGH (85-100%)**: Excellent source match, highly reliable
- **HIGH (70-84%)**: Strong match, good quality
- **MEDIUM (50-69%)**: Moderate match, generally helpful
- **LOW (30-49%)**: Weak match, verify carefully
- **VERY LOW (0-29%)**: Poor match, answer may be unreliable

**Factors:**
- Average similarity score of retrieved sources
- Maximum similarity score
- Number of sources found
- Consistency across sources (low variance)

## ⚙️ Configuration

### Environment Variables

Create `.env` file with:

```env
# Required
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Optional
GOOGLE_API_KEY=...  # For Gemini models
COHERE_API_KEY=...  # For neural reranking
LOG_LEVEL=INFO      # DEBUG, INFO, WARNING, ERROR
```

### Advanced Settings

Edit `rag_gui.py` to customize:

```python
# Line 69: Change embedding model
Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-3-small"  # or text-embedding-3-large
)

# Line 72: Change LLM
Settings.llm = Anthropic(
    model="claude-sonnet-4-5",  # or claude-3-5-sonnet-20241022
    temperature=0  # 0 = deterministic, 1 = creative
)

# Line 205: Adjust cache threshold
self.cache = SemanticCache(
    similarity_threshold=0.92  # 0.90 = more cache hits, 0.95 = more precise
)

# Line 296: Change retrieval parameters
vector_retriever = vector_index.as_retriever(
    similarity_top_k=5  # Number of results per retriever
)
```

## 🔧 Troubleshooting

### GUI Won't Start

**Problem**: Import errors or missing dependencies

**Solution:**
```bash
pip install --upgrade -r requirements.txt
```

### Missing API Keys

**Problem**: "ERROR: Missing API keys!"

**Solution:**
1. Create `.env` file: `cp .env.example .env`
2. Edit `.env` and add your keys
3. Restart the GUI

### Documentation Not Found

**Problem**: "Documentation directory 'docs_bakkesmod_only' not found!"

**Solution:**
Ensure you have the documentation files:
```bash
ls docs_bakkesmod_only/
```

If missing, clone the full repository.

### Slow Responses

**Problem**: Queries take 10+ seconds

**Solutions:**
- Enable semantic cache (should be on by default)
- Check internet connection
- Verify API keys are valid
- Try asking simpler questions

### Code Generation Fails

**Problem**: No code generated or validation errors

**Solutions:**
- Be more specific in requirements
- Mention "BakkesMod plugin" explicitly
- Include specific features (events, ImGui, etc.)
- Check API key has sufficient credits

### Port Already in Use

**Problem**: "Error: Port 7860 already in use"

**Solution:**
Edit `rag_gui.py` line 692:
```python
demo.launch(
    server_port=7861  # Change to different port
)
```

## 📊 Performance

### Typical Metrics

| Metric | Value | Notes |
|--------|-------|-------|
| **Query time (cached)** | 50-100ms | Semantic cache hit |
| **Query time (uncached)** | 1-3s | Full retrieval pipeline |
| **Code generation** | 5-10s | Includes validation |
| **Cost per query** | $0.01-0.05 | Without caching |
| **Cache hit rate** | 30-40% | On typical workloads |

### Optimization Tips

1. **Use caching**: Enable for 30-40% cost savings
2. **Be specific**: Precise questions = better results
3. **Reuse queries**: Similar questions hit cache
4. **Batch questions**: Ask multiple related questions
5. **Check confidence**: High confidence = reliable answer

## 🔐 Security

### API Keys

- **Never commit** `.env` file to git (it's in `.gitignore`)
- **Use environment variables** only
- **Rotate keys** periodically
- **Monitor usage** on provider dashboards

### Data Privacy

- Queries are sent to OpenAI (embeddings) and Anthropic (LLM)
- No data is stored on external servers beyond provider logs
- Cache is local (`.cache/semantic/`)
- Generated code is saved locally only

## 🚢 Deployment

### Local Development

Use the startup scripts as described above.

### Docker Deployment

```bash
docker build -t bakkesmod-rag-gui .
docker run -p 7860:7860 --env-file .env bakkesmod-rag-gui
```

### Production Deployment

For production use:

1. **Use HTTPS**: Deploy behind reverse proxy (nginx, Caddy)
2. **Add authentication**: Protect with basic auth or OAuth
3. **Set rate limits**: Prevent abuse
4. **Monitor costs**: Track API usage
5. **Enable logging**: Use structured logs for debugging

Example with nginx:
```nginx
server {
    listen 443 ssl;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://localhost:7860;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

## 📖 API Documentation

### Programmatic Access

Use the RAG system programmatically:

```python
from rag_gui import BakkesModRAGGUI

# Initialize
app = BakkesModRAGGUI()

# Query
for chunk in app.query_rag("How do I hook events?"):
    print(chunk, end="")

# Generate code
header, impl, status = app.generate_code(
    "Create a plugin that logs boost amount"
)

# Export
result = app.export_code("BoostLogger")
print(result)

# Statistics
stats = app.get_statistics()
print(stats)
```

## 🤝 Contributing

Found a bug? Have a feature request?

1. Check existing issues on GitHub
2. Create a new issue with detailed description
3. Or submit a pull request

## 📝 License

MIT License - see LICENSE file for details

## 🔗 Related Links

- [BakkesMod Official Site](https://bakkesmod.com/)
- [BakkesMod SDK Documentation](https://github.com/bakkesmodorg/BakkesModSDK)
- [LlamaIndex Documentation](https://docs.llamaindex.ai/)
- [Anthropic Claude](https://www.anthropic.com/claude)
- [OpenAI API](https://platform.openai.com/)

## 💡 Tips & Tricks

### Getting Better Answers

1. **Be specific**: "How do I hook the OnGoalScored event?" vs "How do I hook events?"
2. **Include context**: Mention if you're using ImGui, specific classes, etc.
3. **Ask follow-ups**: Build on previous answers
4. **Check sources**: Verify against cited documentation

### Code Generation Best Practices

1. **Start simple**: Generate basic plugin first, then add features
2. **Be descriptive**: Explain what you want clearly
3. **Review code**: Always check generated code before using
4. **Validate**: Use the built-in validation
5. **Test**: Compile and test in BakkesMod

### Cost Optimization

1. **Enable caching**: Saves 30-40% on costs
2. **Reuse queries**: Ask similar questions to hit cache
3. **Be efficient**: One good question > multiple vague ones
4. **Monitor usage**: Check statistics tab regularly

---

**Need help?** Open an issue on GitHub or check the help tab in the GUI!
