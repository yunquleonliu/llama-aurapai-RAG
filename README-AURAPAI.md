# Llama-aurapai-RAG

A specialized fork of [llama.cpp](https://github.com/ggerganov/llama.cpp) with integrated **RAG (Retrieval-Augmented Generation)** middleware for context-augmented AI responses.

🌐 **Live Demo:** [https://aurapai.dpdns.org](https://aurapai.dpdns.org)

## ✨ Additional Features vs Upstream

### 🔍 RAG Middleware Integration
- Built-in middleware for augmenting LLM responses with retrieved context
- Connects to external RAG services (Aurapai or custom implementations)
- Automatic context injection into chat completions
- Configurable similarity thresholds and result limits

### 🌐 Full URL Support for RAG Services
- Support for both `localhost:port` and full URLs (`http://domain.com`)
- HTTPS support with certificate verification
- Flexible deployment options (local or remote RAG services)

### 🎨 Custom Branding
- Rebranded UI as "Llama-aurapai-RAG" for production deployment
- Customized for the Aurapai platform

## 🚀 Quick Start

### Building

```bash
# Clone the repository
git clone https://github.com/yunquleonliu/llama-aurapai-RAG.git
cd llama-aurapai-RAG

# Build with CUDA support
mkdir build-cuda
cd build-cuda
cmake .. -DGGML_CUDA=ON
cmake --build . --config Release --target llama-server
```

### Running with RAG

**With remote RAG service:**
```bash
./build-cuda/bin/llama-server \
  -m /path/to/model.gguf \
  -ngl 99 -c 4096 \
  --host 0.0.0.0 --port 8080 \
  --rag-enabled \
  --rag-host http://your-rag-service.com \
  --rag-max-results 5 \
  --rag-similarity-threshold 0.3
```

**With local RAG service:**
```bash
./build-cuda/bin/llama-server \
  -m /path/to/model.gguf \
  -ngl 99 -c 4096 \
  --host 0.0.0.0 --port 8080 \
  --rag-enabled \
  --rag-host localhost \
  --rag-port 8001 \
  --rag-max-results 5 \
  --rag-similarity-threshold 0.3
```

## 📋 RAG Configuration Options

| Option | Description | Default |
|--------|-------------|---------|
| `--rag-enabled` | Enable RAG middleware | disabled |
| `--rag-host` | RAG service host/URL | `localhost` |
| `--rag-port` | RAG service port (for localhost) | `8001` |
| `--rag-max-results` | Maximum context chunks to retrieve | `5` |
| `--rag-similarity-threshold` | Minimum similarity score (0.0-1.0) | `0.3` |
| `--rag-include-tools` | Include tool suggestions in RAG response | `false` |
| `--rag-timeout-ms` | Request timeout in milliseconds | `5000` |

## 🏗️ Architecture

```
┌─────────────┐
│  llama.cpp  │
│   Server    │
└──────┬──────┘
       │
       │ Chat Request
       ▼
┌──────────────┐
│     RAG      │──────► Query Vector DB
│  Middleware  │◄────── Retrieved Context
└──────┬───────┘
       │
       │ Augmented Prompt
       ▼
┌──────────────┐
│     LLM      │
│  Inference   │
└──────┬───────┘
       │
       │ Response
       ▼
```

## 🔌 RAG Service API

The RAG middleware expects the following HTTP endpoint:

**POST** `/api/v1/llama/augment`

**Request:**
```json
{
  "query": "user query text",
  "max_results": 5,
  "similarity_threshold": 0.3,
  "include_tools": false,
  "session_id": "optional-session-id"
}
```

**Response:**
```json
{
  "success": true,
  "augmented_context": "Retrieved context text...",
  "chunks": [
    {
      "content": "chunk text",
      "metadata": {"source": "doc.pdf", "page": 1},
      "similarity": 0.85
    }
  ],
  "suggested_tools": ["web_search"],
  "metadata": {
    "chunks_found": 3,
    "augmentation_applied": true
  },
  "latency_ms": 125.5
}
```

## 🛠️ Use Cases

1. **Document Q&A**: Augment responses with relevant documentation
2. **Code Assistance**: Inject code examples from repositories
3. **Knowledge Base**: Connect to company wikis or knowledge bases
4. **Multi-modal RAG**: Combine text, code, and structured data

## 📦 Compatible RAG Backends

This fork works with any RAG service that implements the API above. Examples:
- **Aurapai** (reference implementation)
- **LangChain** servers
- **Custom FastAPI** services with ChromaDB/Pinecone/Weaviate

## 🔮 Future Plans

- [ ] Generic RAG provider plugin system
- [ ] Support for multiple RAG backends simultaneously
- [ ] gRPC protocol support
- [ ] Embedded vector database option
- [ ] RAG response streaming
- [ ] Submit generalized version as upstream PR

## 🤝 Contributing

We welcome contributions! Areas of focus:
1. Making the RAG interface more generic
2. Adding support for other vector databases
3. Performance optimizations
4. Documentation improvements

## 📝 License

This project inherits the [MIT License](LICENSE) from llama.cpp.

## 🙏 Acknowledgments

- **[llama.cpp](https://github.com/ggerganov/llama.cpp)**: Base inference engine by Georgi Gerganov
- **Aurapai Team**: RAG service implementation and testing
- **Community**: Feedback and testing

## 🔗 Links

- **Upstream**: [ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp)
- **Live Demo**: [https://aurapai.dpdns.org](https://aurapai.dpdns.org)
- **Issues**: [GitHub Issues](https://github.com/yunquleonliu/llama-aurapai-RAG/issues)

---

**Note**: This is a specialized fork. For general llama.cpp usage, please refer to the [upstream repository](https://github.com/ggerganov/llama.cpp).
