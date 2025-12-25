# RAG System - PDF Document Retrieval

A Retrieval Augmented Generation (RAG) system that loads PDF documents from AWS S3, generates embeddings, and enables semantic search using ChromaDB vector store.

## 🚀 Features

- **S3 PDF Loading**: Automatically loads and processes PDF files from AWS S3 buckets
- **Document Chunking**: Intelligently splits documents into manageable chunks with overlap
- **Vector Embeddings**: Uses Sentence Transformers (all-MiniLM-L6-v2) for high-quality embeddings
- **Persistent Vector Store**: ChromaDB for efficient similarity search and persistence
- **Semantic Search**: Query documents using natural language with similarity scoring
- **LLM Integration**: Complete RAG pipeline with Hugging Face open source models (FLAN-T5) for question answering

## 📋 Prerequisites

- Python 3.13+
- AWS Account with S3 access
- AWS credentials configured (`~/.aws/credentials` or environment variables)
- S3 bucket with PDF files

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/cmuthukumar/rag-pdf-search.git
   cd rag-pdf-search
   ```

2. **Create and activate virtual environment**
   ```bash
   python3.13 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 📦 Dependencies

- LangChain (document processing)
- ChromaDB (vector storage)
- Sentence Transformers (embeddings)
- PyPDF & PyMuPDF (PDF parsing)
- Boto3 (AWS S3 integration)
- NumPy & scikit-learn (ML utilities)
- Transformers & PyTorch (LLM inference)

## 🔧 Configuration

### AWS Credentials

Ensure your AWS credentials are configured:

```bash
# Option 1: AWS CLI
aws configure

# Option 2: Environment variables
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_DEFAULT_REGION=us-east-1
```

### S3 Bucket Setup

Update the bucket name and prefix in the notebook:

```python
doc_processor = DocumentProcessor(
    bucket_name="your-bucket-name",  # Change this
    prefix="pdfs/"                    # Change this if needed
)
```

## 📓 Notebook

### `s3pdf_loader_to_vector_store.ipynb`
Complete RAG pipeline for S3 PDFs.

**Features:**
- Load PDFs from S3 using PyPDFLoader
- Split documents into chunks (1500 chars with 50 char overlap)
- Generate embeddings using Sentence Transformers
- Store in ChromaDB vector database
- Semantic search with similarity scoring

**Usage:**
```bash
jupyter notebook notebook/s3pdf_loader_to_vector_store.ipynb
```

## 🎯 Quick Start

### Step 1: Start Jupyter Notebook

```bash
# Activate virtual environment
source .venv/bin/activate

# Start Jupyter
jupyter notebook
```

### Step 2: Run the Notebook

1. Open `notebook/s3pdf_loader_to_vector_store.ipynb`
2. Update the S3 bucket configuration
3. Run cells sequentially:

```python
# Cell 1: Load documents from S3
doc_processor = DocumentProcessor(bucket_name="your-bucket", prefix="pdfs/")
all_documents = doc_processor.load_documents()

# Cell 2: Split into chunks
document_chunks = doc_processor.split_documents_into_chunks(
    chunk_size=1500, 
    chunk_overlap=50
)

# Cell 3: Generate embeddings
texts = [doc.page_content for doc in document_chunks]
embeddings = embedding_manager.generate_embedding(texts)

# Cell 4: Store in vector database
vectorStore.add_documents(document_chunks, embeddings)

# Cell 5: Query the system
results = rag_retriever.retrieve(
    query="your search query here",
    top_k=5,
    score_threshold=0.4
)
```

### Step 3: Query Your Documents

#### Option A: Semantic Search Only
```python
# Search for relevant documents
results = rag_retriever.retrieve(
    query="your search query here",
    top_k=5,              # Number of results to return
    score_threshold=0.4   # Minimum similarity score (0.0-1.0)
)

# View results
for doc in results:
    print(f"Source: {doc['metadata']['source_file']}")
    print(f"Page: {doc['metadata']['page']}")
    print(f"Score: {doc['similarity_score']:.4f}")
    print(f"Content: {doc['content'][:200]}...\n")
```

#### Option B: RAG with LLM (Question Answering)
```python
# Initialize RAG with LLM
rag_llm = RAGWithLLM(
    retriever=rag_retriever,
    model_name="google/flan-t5-base"  # or "google/flan-t5-small" for faster inference
)

# Ask questions in natural language
result = rag_llm.generate_answer(
    query="What is the main topic discussed in the documents?",
    top_k=3,
    score_threshold=0.3,
    max_length=256
)

# View the answer
print(f"Question: {result['query']}")
print(f"Answer: {result['answer']}")
print(f"Sources: {result['num_sources']} documents used")

# View detailed sources
for i, source in enumerate(result['sources'], 1):
    print(f"\n{i}. {source['source_file']} (Page {source['page']})")
    print(f"   Similarity: {source['similarity_score']:.4f}")
```

## 📊 System Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   AWS S3    │────▶│  PyPDFLoader │────▶│   Chunks    │
│   (PDFs)    │     │  (Extract)   │     │  (Split)    │
└─────────────┘     └──────────────┘     └─────────────┘
                                                │
                                                ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Query     │────▶│  Embeddings  │────▶│  ChromaDB   │
│  (Search)   │     │   (Model)    │     │   (Store)   │
└─────────────┘     └──────────────┘     └─────────────┘
                            │                   │
                            └────────┬──────────┘
                                     ▼
                            ┌──────────────┐
                            │ RAG Retriever│
                            │  (Top-K)     │
                            └──────────────┘
                                     │
                                     ▼
                            ┌──────────────┐
                            │  LLM (FLAN-T5)│────▶ Answer + Sources
                            │  (Generate)  │
                            └──────────────┘
```

## 🔍 How It Works

1. **Document Loading**: Downloads PDFs from S3, extracts text using PyPDFLoader
2. **Chunking**: Splits documents using RecursiveCharacterTextSplitter with overlap
3. **Embedding Generation**: Uses `all-MiniLM-L6-v2` model (384-dimensional embeddings)
4. **Vector Storage**: Stores embeddings in ChromaDB with metadata
5. **Retrieval**: Finds top-k most similar chunks using L2 distance
6. **Answer Generation**: (Optional) Uses FLAN-T5 LLM to generate natural language answers from retrieved context

## ⚙️ Configuration Options

### Chunk Size
```python
document_chunks = doc_processor.split_documents_into_chunks(
    chunk_size=1500,     # Adjust based on your content
    chunk_overlap=50     # Increase for more context
)
```

### Embedding Model
```python
embedding_manager = EmbeddingManager(
    model_name='all-MiniLM-L6-v2'  # Fast, good quality
    # model_name='all-mpnet-base-v2'  # Slower, better quality
)
```

### Search Parameters
```python
results = rag_retriever.retrieve(
    query="your query",
    top_k=5,              # More results = more context
    score_threshold=0.4   # Higher = more strict
)
```

### Similarity Score Guide
| Threshold | Description | Use Case |
|-----------|-------------|----------|
| 0.7+ | Very strict | Exact matches only |
| 0.4-0.7 | Balanced | Most use cases |
| 0.1-0.4 | Permissive | Exploratory search |

### LLM Model Selection
```python
# FLAN-T5 Small - Fast inference, lower quality
rag_llm = RAGWithLLM(retriever=rag_retriever, model_name="google/flan-t5-small")

# FLAN-T5 Base - Balanced (recommended)
rag_llm = RAGWithLLM(retriever=rag_retriever, model_name="google/flan-t5-base")

# FLAN-T5 Large - Best quality, slower
rag_llm = RAGWithLLM(retriever=rag_retriever, model_name="google/flan-t5-large")
```

| Model | Size | Speed | Quality | Use Case |
|-------|------|-------|---------|----------|
| flan-t5-small | 80M | Fast | Good | Development/testing |
| flan-t5-base | 250M | Medium | Better | Production (balanced) |
| flan-t5-large | 780M | Slow | Best | High-quality answers |

## 📁 Project Structure

```
rag-pdf-search/
├── notebook/
│   └── s3pdf_loader_to_vector_store.ipynb  # Full RAG pipeline
├── data/
│   └── vector_store/                       # ChromaDB storage (auto-generated)
├── requirements.txt                        # Python dependencies
├── pyproject.toml                          # Project metadata
├── .gitignore                             # Git ignore rules
├── CONTRIBUTING.md                        # Contribution guidelines
├── GITHUB_SETUP.md                        # GitHub setup instructions
└── README.md                              # This file
```

## 🐛 Troubleshooting

### Issue: "No module named 'langchain.document_loaders'"
**Solution:** Import from correct module:
```python
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
```

### Issue: "No documents retrieved"
**Solutions:**
1. Lower the `score_threshold` (try 0.1)
2. Check if documents were added to vector store
3. Verify embeddings were generated correctly

### Issue: AWS credentials error
**Solution:** Configure AWS credentials:
```bash
aws configure
# Or set environment variables
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
```

### Issue: Memory issues with large PDFs
**Solution:** Reduce chunk size:
```python
chunk_size=500  # Smaller chunks
```

### Issue: LLM loading errors or out of memory
**Solutions:**
1. Use smaller model: `model_name="google/flan-t5-small"`
2. For CPU-only systems, FLAN-T5 base works well
3. Ensure you have sufficient RAM (4GB+ recommended)
4. Install torch: `pip install torch transformers`

### Issue: LLM answers are generic or incorrect
**Solutions:**
1. Lower `score_threshold` to retrieve more context
2. Increase `top_k` to get more documents
3. Adjust `chunk_size` for better context chunks
4. Try a larger model (flan-t5-large)

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📝 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- [LangChain](https://github.com/langchain-ai/langchain) for document processing
- [ChromaDB](https://github.com/chroma-core/chroma) for vector storage
- [Sentence Transformers](https://github.com/UKPLab/sentence-transformers) for embeddings
- [Hugging Face](https://huggingface.co/) for FLAN-T5 models and Transformers library
- [Google Research](https://github.com/google-research/t5x) for FLAN-T5 instruction-tuned models

