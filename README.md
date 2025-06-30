# 🔬 Local Research Assistant

A powerful RAG-powered research paper analysis tool that runs entirely on your local machine using Ollama and open-source language models. Upload PDF research papers and get intelligent analysis, summaries, and answers to your questions without sending data to external services.


## ✨ Features

### 🎯 Core Capabilities
- **Local AI Processing**: Uses Ollama for completely private, offline AI analysis
- **Multi-Document RAG**: Retrieval-Augmented Generation across multiple research papers
- **Intelligent PDF Processing**: Advanced text extraction with table and metadata recognition
- **Smart Chunking**: Context-aware document segmentation for optimal retrieval
- **Interactive Chat**: Natural language querying of your research documents
- **Citation Tracking**: Automatic source attribution with page references

### 📊 Analysis Tools
- **Document Summaries**: AI-generated comprehensive paper summaries
- **Keyword Extraction**: Automated topic and keyword identification
- **Table Visualization**: Automatic chart generation from extracted tables
- **Cross-Document Search**: Find relevant information across your entire document collection

### 🔒 Privacy & Performance
- **100% Local**: No data leaves your machine
- **Model Flexibility**: Support for multiple Ollama models (Gemma, Llama, Mistral, etc.)
- **Efficient Processing**: TF-IDF based similarity search for fast retrieval
- **Resource Optimized**: Designed for local hardware constraints

## 🚀 Quick Start

### Prerequisites

1. **Python 3.8+** installed on your system
2. **Ollama** installed and running

### Installation

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Install and setup Ollama:**
   
   For Windows: Download from ([Official Ollama Website](https://ollama.com/download/windows))
 

3. **Open and start Ollama server:**
   ```bash
   ollama serve
   ```

4. **Pull a language model:**
   ```bash
   #Recommended model
   ollama run gemma3:4b        
   
   ```

5. **Launch the application:**
   ```bash
   streamlit run research_assistant_localllm.py
   ```

6. **Open your browser:** Navigate to `http://localhost:8501`

## 📖 Usage Guide

### Getting Started

1. **Select Model**: Choose your preferred Ollama model from the sidebar
2. **Upload Papers**: Upload one or more PDF research papers
3. **Process Documents**: Click "Process Documents" to analyze and index your papers
4. **Start Analyzing**: Use the chat interface to ask questions about your research

### Key Features

#### 💬 Chat Interface
Ask natural language questions about your research papers:
- "What are the key findings in these papers?"
- "How do the methodologies compare across studies?"
- "What limitations were identified?"

#### 📊 Charts & Tables
- Automatically extracts tables from PDFs
- Generates interactive visualizations
- Provides data insights and explanations

#### 📝 Document Summaries
- AI-generated comprehensive summaries
- Covers methodology, findings, and conclusions
- Maintains context across multiple papers

#### 🔍 Keywords & Topics
- Extracts key terms and concepts
- Frequency analysis and visualization
- Topic modeling across documents

## 🛠️ Technical Architecture

### Components

- **Document Processor**: PDF text extraction and intelligent chunking
- **Embedding Manager**: TF-IDF based similarity search (no external APIs needed)
- **Local LLM Interface**: Ollama integration for AI processing
- **Chart Generator**: Automated visualization from extracted tables
- **Streamlit UI**: Interactive web interface

### Data Flow

1. PDF Upload → Text Extraction → Intelligent Chunking
2. TF-IDF Vectorization → Similarity Index Creation
3. User Query → Similarity Search → Context Retrieval
4. Local LLM Processing → Response Generation → Citation Tracking

## 📋 Requirements

### System Requirements
- **RAM**: 8GB minimum (16GB recommended for larger models)
- **Storage**: 2GB+ free space bassed on models
- **CPU**: Modern multi-core processor recommended

