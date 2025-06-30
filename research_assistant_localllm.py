import streamlit as st
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
import re
import json
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64
import io
import tempfile
import os
from pathlib import Path

# Local LLM integration
import ollama  # pip install ollama

# PDF processing
import PyPDF2
import pdfplumber
import fitz  # PyMuPDF

# Text processing and embeddings - using TF-IDF instead of sentence transformers
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Additional utilities
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

def check_ollama_models():
    """Check available Ollama models"""
    try:
        models = ollama.list()
        model_names = [model['model'] for model in models['models']]
        return model_names
    except Exception as e:
        st.error(f"Error connecting to Ollama: {str(e)}")
        return None


class DocumentProcessor:
    """Handles PDF processing, text extraction, and chunking"""
    
    def __init__(self):
        self.documents = []
        self.chunks = []
        self.metadata = []
    
    def extract_text_from_pdf(self, pdf_file) -> Dict[str, Any]:
        """Extract text, tables, and metadata from PDF"""
        doc_data = {
            'filename': pdf_file.name,
            'pages': [],
            'tables': [],
            'metadata': {}
        }
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(pdf_file.read())
            tmp_path = tmp_file.name
        
        try:
            # Extract text using PyMuPDF for better formatting
            doc = fitz.open(tmp_path)
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                
                # Extract tables using pdfplumber
                try:
                    with pdfplumber.open(tmp_path) as pdf:
                        if page_num < len(pdf.pages):
                            page_tables = pdf.pages[page_num].extract_tables()
                            if page_tables:
                                doc_data['tables'].extend([
                                    {'page': page_num + 1, 'table': table} 
                                    for table in page_tables if table
                                ])
                except Exception as e:
                    st.warning(f"Could not extract tables from page {page_num + 1}: {str(e)}")
                
                doc_data['pages'].append({
                    'page_number': page_num + 1,
                    'text': text,
                    'char_count': len(text)
                })
            
            # Extract metadata
            metadata = doc.metadata or {}
            doc_data['metadata'] = {
                'title': metadata.get('title', '') or '',
                'author': metadata.get('author', '') or '',
                'subject': metadata.get('subject', '') or '',
                'creator': metadata.get('creator', '') or '',
                'total_pages': len(doc)
            }
            
            doc.close()
            
        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")
        finally:
            # Clean up temporary file
            try:
                os.unlink(tmp_path)
            except:
                pass
        
        return doc_data
    
    def intelligent_chunking(self, text: str, max_chunk_size: int = 1000) -> List[str]:
        """Chunk text intelligently based on sentences and paragraphs"""
        # Split by paragraphs first
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            # Clean paragraph
            paragraph = paragraph.strip()
            if not paragraph:
                continue
                
            # If paragraph is too long, split by sentences
            if len(paragraph) > max_chunk_size:
                try:
                    sentences = sent_tokenize(paragraph)
                    for sentence in sentences:
                        if len(current_chunk + sentence) <= max_chunk_size:
                            current_chunk += sentence + " "
                        else:
                            if current_chunk.strip():
                                chunks.append(current_chunk.strip())
                            current_chunk = sentence + " "
                except:
                    # Fallback to simple splitting if nltk fails
                    sentences = paragraph.split('. ')
                    for sentence in sentences:
                        if len(current_chunk + sentence) <= max_chunk_size:
                            current_chunk += sentence + ". "
                        else:
                            if current_chunk.strip():
                                chunks.append(current_chunk.strip())
                            current_chunk = sentence + ". "
            else:
                if len(current_chunk + paragraph) <= max_chunk_size:
                    current_chunk += paragraph + "\n\n"
                else:
                    if current_chunk.strip():
                        chunks.append(current_chunk.strip())
                    current_chunk = paragraph + "\n\n"
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return [chunk for chunk in chunks if len(chunk.strip()) > 50]  # Filter very short chunks
    
    def process_document(self, pdf_file) -> Dict[str, Any]:
        """Process a single PDF document"""
        doc_data = self.extract_text_from_pdf(pdf_file)
        
        # Combine all text
        full_text = "\n\n".join([page['text'] for page in doc_data['pages']])
        
        # Create chunks
        chunks = self.intelligent_chunking(full_text)
        
        # Create metadata for each chunk
        chunk_metadata = []
        for i, chunk in enumerate(chunks):
            # Find which page this chunk likely belongs to
            page_num = self._find_chunk_page(chunk, doc_data['pages'])
            
            chunk_metadata.append({
                'chunk_id': i,
                'document': doc_data['filename'],
                'page_number': page_num,
                'char_count': len(chunk),
                'word_count': len(chunk.split())
            })
        
        return {
            'document_data': doc_data,
            'chunks': chunks,
            'chunk_metadata': chunk_metadata
        }
    
    def _find_chunk_page(self, chunk: str, pages: List[Dict]) -> int:
        """Find the most likely page number for a chunk"""
        chunk_words = set(chunk.lower().split()[:20])  # First 20 words
        
        best_page = 1
        best_score = 0
        
        for page in pages:
            page_words = set(page['text'].lower().split())
            overlap = len(chunk_words.intersection(page_words))
            
            if overlap > best_score:
                best_score = overlap
                best_page = page['page_number']
        
        return best_page

class SimpleEmbeddingManager:
    """Handles text embeddings using TF-IDF (simpler alternative to sentence transformers)"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            max_df=0.8,
            min_df=2
        )
        self.chunk_vectors = None
        self.chunks = []
        self.metadata = []
    
    def create_embeddings(self, chunks: List[str], metadata: List[Dict]):
        """Create TF-IDF embeddings for text chunks"""
        self.chunks = chunks
        self.metadata = metadata
        
        if not chunks:
            return None
        
        try:
            # Create TF-IDF vectors
            self.chunk_vectors = self.vectorizer.fit_transform(chunks)
            st.success(f"Created embeddings for {len(chunks)} chunks")
            return self.chunk_vectors
        except Exception as e:
            st.error(f"Error creating embeddings: {str(e)}")
            return None
    
    def search_similar_chunks(self, query: str, k: int = 5) -> List[Dict]:
        """Search for similar chunks using TF-IDF cosine similarity"""
        if self.chunk_vectors is None or not self.chunks:
            return []
        
        try:
            # Transform query using the same vectorizer
            query_vector = self.vectorizer.transform([query])
            
            # Calculate cosine similarity
            similarities = cosine_similarity(query_vector, self.chunk_vectors).flatten()
            
            # Get top k results
            top_indices = similarities.argsort()[-k:][::-1]
            
            results = []
            for idx in top_indices:
                if idx < len(self.chunks) and similarities[idx] > 0:
                    results.append({
                        'chunk': self.chunks[idx],
                        'metadata': self.metadata[idx],
                        'similarity_score': float(similarities[idx])
                    })
            
            return results
        except Exception as e:
            st.error(f"Error in similarity search: {str(e)}")
            return []


# CHANGE 2: Modify the LocalGemmaLLM class constructor
class LocalGemmaLLM:
    """Handles local LLM model interactions via Ollama - Connect On Demand"""
    
    def __init__(self, model_name: str):
        """Initialize with user-selected model name - no default model
        
        Args:
            model_name: Name of the model in Ollama (must be provided by user)
        """
        self.model_name = model_name
        self.chat_history = []
        # No connection testing here - we'll connect when needed
    
    def generate_answer(self, query: str, context_chunks: List[Dict], 
                       chat_history: List[Dict] = None) -> Dict[str, Any]:
        """Generate answer using local Gemma 4B model with context"""
        
        if not context_chunks:
            return {
                'answer': "I couldn't find relevant information in the uploaded documents to answer your question.",
                'citations': [],
                'context_used': [],
                'status': 'no_context'
            }
        
        # Prepare context
        context_text = "\n\n".join([
            f"[Source: {chunk['metadata']['document']}, Page {chunk['metadata']['page_number']}]\n{chunk['chunk'][:800]}..."
            for chunk in context_chunks[:3]
        ])
        
        # Prepare chat history context
        history_context = ""
        if chat_history:
            recent_history = chat_history[-2:]
            history_context = "\n".join([
                f"Previous Q: {item['question'][:150]}...\nPrevious A: {item['answer'][:150]}..."
                for item in recent_history
            ])
        
        # Create system message and user message
        system_message = """You are a research assistant analyzing academic papers. Based on the research excerpts provided, answer questions comprehensively and accurately.

Instructions:
- Provide detailed, well-structured answers based on the research context
- Include citations as [Source: filename, Page X]
- If information is insufficient, state this clearly
- Keep responses focused and under 400 words"""
        
        user_message = f"""Research Context:
{context_text}

{f"Previous context: {history_context}" if history_context else ""}

Question: {query}

Please provide a comprehensive answer based on the research context above."""
        
        try:
            # This is where the connection happens - only when needed!
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ]
            )
            
            # Extract the response content
            answer = response['message']['content']
            
            if not answer:
                return {
                    'answer': "I received an empty response. Please try rephrasing your question.",
                    'citations': [],
                    'context_used': context_chunks,
                    'status': 'empty_response'
                }
            
            # Extract citations
            citations = self._extract_citations(answer, context_chunks)
            
            return {
                'answer': answer,
                'citations': citations,
                'context_used': context_chunks,
                'status': 'success'
            }
            
        except Exception as e:
            # This is where connection errors will be caught and handled gracefully
            error_msg = str(e).lower()
            if "connection refused" in error_msg or "ollama" in error_msg:
                return {
                    'answer': f"""❌ **Ollama Connection Error**

I couldn't connect to your local Ollama server. Please make sure:

1. **Ollama is installed:** Download from https://ollama.ai
2. **Ollama server is running:** Run `ollama serve` in your terminal
3. **Gemma model is available:** Run `ollama pull {self.model_name}`

**Quick Setup:**
```bash
# Install Ollama (if not installed)
curl -fsSL https://ollama.ai/install.sh | sh

# Start the server
ollama serve

# Pull the model (in another terminal)
ollama pull {self.model_name}
```

Once Ollama is running, try your question again!""",
                    'citations': [],
                    'context_used': [],
                    'status': 'connection_error'
                }
            else:
                return {
                    'answer': f"Error generating response: {str(e)}",
                    'citations': [],
                    'context_used': [],
                    'status': 'error'
                }
    
    def _extract_citations(self, text: str, context_chunks: List[Dict]) -> List[Dict]:
        """Extract citation information from generated text"""
        citations = []
        
        # Look for citation patterns
        citation_pattern = r'\[Source: ([^,]+), Page (\d+)\]'
        matches = re.findall(citation_pattern, text)
        
        for filename, page in matches:
            # Find the corresponding chunk
            for chunk in context_chunks:
                if (chunk['metadata']['document'] == filename and 
                    chunk['metadata']['page_number'] == int(page)):
                    citations.append({
                        'document': filename,
                        'page': int(page),
                        'chunk_preview': chunk['chunk'][:200] + "...",
                        'similarity_score': chunk['similarity_score']
                    })
                    break
        
        return citations
    
    def summarize_document(self, chunks: List[str], metadata: Dict) -> str:
        """Generate document summary using 4B model"""
        if not chunks:
            return "No content available for summary."
        
        # Use more chunks for summary
        summary_chunks = chunks[:4]
        combined_text = "\n\n".join(summary_chunks)
        
        # Limit text length for 4B model
        if len(combined_text) > 3000:
            combined_text = combined_text[:3000] + "..."
        
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a research assistant that creates comprehensive summaries of academic papers."
                    },
                    {
                        "role": "user", 
                        "content": f"""Summarize this research paper excerpt comprehensively:

Document: {metadata.get('title', 'Research Paper')}
Author: {metadata.get('author', 'Unknown')}

Content:
{combined_text}

Provide a detailed summary covering:
1. Main research objective and background
2. Methodology used
3. Key findings and results
4. Conclusions and implications
5. Limitations and future work

Keep the summary under 300 words but be comprehensive."""
                    }
                ]
            )
            return response['message']['content'] if response['message']['content'] else "Could not generate summary."
        except Exception as e:
            error_msg = str(e).lower()
            if "connection refused" in error_msg or "ollama" in error_msg:
                return f"❌ **Connection Error:** Please make sure Ollama is running (`ollama serve`) and the model `{self.model_name}` is available (`ollama pull {self.model_name}`)."
            return f"Error generating summary: {str(e)}"
    
    def extract_keywords(self, text: str) -> List[str]:
        """Extract keywords using NLP techniques"""
        try:
            # Simple keyword extraction using TF-IDF
            stop_words = set(stopwords.words('english'))
            words = word_tokenize(text.lower())
            words = [word for word in words if word.isalpha() and word not in stop_words and len(word) > 3]
            
            # Get most frequent words
            word_freq = Counter(words)
            keywords = [word for word, count in word_freq.most_common(20)]
            
            return keywords
        except:
            # Fallback method
            words = text.lower().split()
            word_freq = Counter([word for word in words if len(word) > 4])
            return [word for word, count in word_freq.most_common(15)]

class ChartGenerator:
    """Handles chart generation from extracted tables"""
    
    @staticmethod
    def create_chart_from_table(table_data: List[List[str]], title: str = "Data Visualization"):
        """Create appropriate chart from table data"""
        if not table_data or len(table_data) < 2:
            return None
        
        try:
            # Clean table data
            cleaned_data = []
            for row in table_data:
                if row and any(cell for cell in row if cell):  # Skip empty rows
                    cleaned_data.append([str(cell) if cell else "" for cell in row])
            
            if len(cleaned_data) < 2:
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(cleaned_data[1:], columns=cleaned_data[0])
            df = df.dropna(how='all')  # Drop completely empty rows
            
            if df.empty:
                return None
            
            # Determine chart type based on data
            numeric_cols = []
            for col in df.columns:
                try:
                    # Try to convert to numeric
                    numeric_series = pd.to_numeric(df[col], errors='coerce')
                    if not numeric_series.isna().all():
                        df[col] = numeric_series
                        numeric_cols.append(col)
                except:
                    continue
            
            if len(numeric_cols) == 0:
                return None
            
            # Create chart based on data structure
            if len(df.columns) >= 2 and len(numeric_cols) >= 1:
                # Bar chart
                x_col = df.columns[0]
                y_col = numeric_cols[0]
                fig = px.bar(df.head(10), x=x_col, y=y_col, title=title)
                fig.update_layout(xaxis_tickangle=-45)
                return fig
            
        except Exception as e:
            st.warning(f"Could not create chart: {str(e)}")
            return None
    
    @staticmethod
    def explain_chart(table_data: List[List[str]], chart_type: str = "bar") -> str:
        """Generate explanation for the chart"""
        if not table_data or len(table_data) < 2:
            return "No data available for explanation."
        
        rows = len(table_data) - 1  # Exclude header
        cols = len(table_data[0]) if table_data[0] else 0
        
        explanation = f"""This {chart_type} chart displays data from a table with {rows} data rows and {cols} columns. 
The visualization helps identify patterns and relationships in the numerical data extracted from the research paper."""
        
        return explanation

def main():
    st.set_page_config(
        page_title="Local Research Assistant",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🔬 Local Research Assistant")
    st.markdown("*RAG-powered research paper analysis with Local LLM*")
    
    # Initialize session state
    if 'doc_processor' not in st.session_state:
        st.session_state.doc_processor = DocumentProcessor()
    if 'embedding_manager' not in st.session_state:
        st.session_state.embedding_manager = SimpleEmbeddingManager()
    # REMOVED: Don't initialize gemma_llm here anymore
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'documents' not in st.session_state:
        st.session_state.documents = []
    if 'processed_chunks' not in st.session_state:
        st.session_state.processed_chunks = []
    if 'chunk_metadata' not in st.session_state:
        st.session_state.chunk_metadata = []
    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = None
    if 'available_models' not in st.session_state:
        st.session_state.available_models = None
    
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model selection - NEW SECTION
        st.subheader("🤖 Model Selection")
        
        # Check for available models
        if st.session_state.available_models is None:
            with st.spinner("Checking available Ollama models..."):
                st.session_state.available_models = check_ollama_models()
        
        if st.session_state.available_models is None:
            st.error("❌ Cannot connect to Ollama server")
            st.markdown("""
            **Please ensure Ollama is running:**
            ```bash
            ollama serve
            ```
            """)
            if st.button("🔄 Retry Connection"):
                st.session_state.available_models = None
                st.rerun()
        
        elif len(st.session_state.available_models) == 0:
            st.warning("⚠️ No models found in Ollama")
            st.markdown("""
            **Install a model first:**
            ```bash
            ollama pull gemma3:4b-it-qat
            # or
            ollama pull llama2
            ```
            """)
            if st.button("🔄 Refresh Models"):
                st.session_state.available_models = None
                st.rerun()
        
        else:
            st.success(f"✅ Found {len(st.session_state.available_models)} models")
            
            # Model selection dropdown
            selected_model = st.selectbox(
                "Select Model:",
                st.session_state.available_models,
                index=None,  # No default selection
                help="Choose which Ollama model to use for analysis"
            )
            
            if selected_model and selected_model != st.session_state.selected_model:
                st.session_state.selected_model = selected_model
                # Initialize LLM with user-selected model (no default)
                st.session_state.gemma_llm = LocalGemmaLLM(model_name=selected_model)
                st.success(f"🎯 Selected: {selected_model}")
            elif selected_model:
                st.info(f"✅ Using: {selected_model}")
        
        st.divider()
        
        st.header("📁 Document Management")
        
        # Only show file upload if model is selected
        if st.session_state.selected_model:
            # File upload
            uploaded_files = st.file_uploader(
                "Upload Research Papers (PDF)",
                type=['pdf'],
                accept_multiple_files=True,
                help="Upload one or more research papers in PDF format"
            )
            
            if uploaded_files:
                if st.button("🔄 Process Documents"):
                    process_documents(uploaded_files)
        else:
            st.info("👆 Please select a model first to enable document upload")
        
        # Document list (rest remains the same)
        if st.session_state.documents:
            st.subheader("📋 Processed Documents")
            for i, doc in enumerate(st.session_state.documents):
                with st.expander(f"📄 {doc['document_data']['filename']}"):
                    st.write(f"**Pages:** {doc['document_data']['metadata']['total_pages']}")
                    st.write(f"**Chunks:** {len(doc['chunks'])}")
                    if doc['document_data']['metadata'].get('title'):
                        st.write(f"**Title:** {doc['document_data']['metadata']['title']}")
                    if doc['document_data']['metadata'].get('author'):
                        st.write(f"**Author:** {doc['document_data']['metadata']['author']}")

    # CHANGE 5: Update the main content area condition
    # Main content area
    if not st.session_state.selected_model:
        st.info("🤖 Please select an Ollama model from the sidebar to begin.")
        
        # Show setup instructions in an expandable section
        with st.expander("🔧 Setup Instructions for Local Ollama"):
            st.markdown("""
            **This app uses your local Ollama server for AI processing. Here's how to set it up:**
            
            ### 1. Install Ollama
            ```bash
            # On macOS/Linux
            curl -fsSL https://ollama.ai/install.sh | sh
            
            # On Windows: Download from https://ollama.ai
            ```
            
            ### 2. Start Ollama Server
            ```bash
            ollama serve
            ```
            
            ### 3. Pull Some Models
            ```bash
            ollama pull gemma3:4b-it-qat
            ollama pull llama2
            ollama pull mistral
            ```
            
            ### 4. Verify Installation
            ```bash
            ollama list  # Should show your installed models
            ```
            
            **Once models are installed**, refresh this page and select a model from the sidebar.
            """)
    
    elif not st.session_state.documents:
        st.info(f"📤 Model '{st.session_state.selected_model}' selected. Upload some PDF research papers to begin analysis.")
    
    else:
        # Show current model info
        st.success(f"🤖 Using model: **{st.session_state.selected_model}**")
        
        # Main interface tabs (rest remains the same)
        tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat", "📊 Charts & Tables", "📝 Summaries", "🔍 Keywords"])
        
        with tab1:
            chat_interface()
        
        with tab2:
            charts_interface()
        
        with tab3:
            summaries_interface()
        
        with tab4:
            keywords_interface()

def process_documents(uploaded_files):
    """Process uploaded PDF documents"""
    with st.spinner("Processing documents..."):
        progress_bar = st.progress(0)
        all_chunks = []
        all_metadata = []
        
        for i, pdf_file in enumerate(uploaded_files):
            try:
                # Reset file pointer
                pdf_file.seek(0)
                
                # Process document
                result = st.session_state.doc_processor.process_document(pdf_file)
                
                if result['chunks']:
                    st.session_state.documents.append(result)
                    
                    # Collect chunks and metadata
                    all_chunks.extend(result['chunks'])
                    all_metadata.extend(result['chunk_metadata'])
                    
                    st.success(f"✅ Processed {pdf_file.name}: {len(result['chunks'])} chunks")
                else:
                    st.warning(f"⚠️ No content extracted from {pdf_file.name}")
                
            except Exception as e:
                st.error(f"❌ Error processing {pdf_file.name}: {str(e)}")
            
            progress_bar.progress((i + 1) / len(uploaded_files))
        
        if all_chunks:
            # Create embeddings
            with st.spinner("Creating embeddings..."):
                st.session_state.embedding_manager.create_embeddings(all_chunks, all_metadata)
                st.session_state.processed_chunks = all_chunks
                st.session_state.chunk_metadata = all_metadata
            
            st.success(f"🎉 Successfully processed {len(st.session_state.documents)} documents with {len(all_chunks)} total chunks!")
        else:
            st.error("❌ No content was extracted from any documents. Please check your PDF files.")

def chat_interface():
    """Main chat interface"""
    st.header("💬 Research Chat")
    
    # Display chat history
    if st.session_state.chat_history:
        for i, chat in enumerate(st.session_state.chat_history):
            with st.container():
                st.markdown(f"**🙋 You:** {chat['question']}")
                st.markdown(f"**🤖 Assistant:** {chat['answer']}")
                
                # Show citations
                if chat.get('citations'):
                    with st.expander("📚 Sources"):
                        for citation in chat['citations']:
                            st.write(f"**{citation['document']}** (Page {citation['page']})")
                            st.write(f"*Similarity: {citation['similarity_score']:.3f}*")
                            st.write(f"Preview: {citation['chunk_preview']}")
                            st.divider()
                
                st.divider()
    
    # Chat input
    query = st.text_input("Ask a question about your research papers:", 
                         placeholder="What are the main findings in the uploaded papers?",
                         key="chat_input")
    
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("🚀 Ask", type="primary"):
            if query:
                handle_query(query)
            else:
                st.warning("Please enter a question.")
    
    with col2:
        if st.button("🗑️ Clear History"):
            st.session_state.chat_history = []
            st.rerun()

def handle_query(query: str):
    """Handle user query and generate response"""
    
    # Check if model is selected and LLM is initialized
    if not st.session_state.selected_model or 'gemma_llm' not in st.session_state:
        st.error("❌ No model selected. Please select a model from the sidebar.")
        return
    
    with st.spinner("Searching and analyzing..."):
        # Find relevant chunks
        similar_chunks = st.session_state.embedding_manager.search_similar_chunks(query, k=5)
        
        if not similar_chunks:
            st.warning("No relevant content found in the uploaded documents. Try rephrasing your question.")
            return
        
        # Generate answer - connection happens here automatically
        result = st.session_state.gemma_llm.generate_answer(
            query, similar_chunks, st.session_state.chat_history
        )
        
        if result['status'] in ['success', 'no_context']:
            # Add to chat history
            st.session_state.chat_history.append({
                'question': query,
                'answer': result['answer'],
                'citations': result['citations'],
                'timestamp': datetime.now().isoformat(),
                'model_used': st.session_state.selected_model  # Track which model was used
            })
            
            st.rerun()
        else:
            # Show the error (which includes helpful setup instructions)
            st.error("Connection Issue")
            st.markdown(result['answer'])  # This contains the formatted error message

def charts_interface():
    """Interface for viewing charts and tables"""
    st.header("📊 Charts & Tables")
    
    # Extract all tables from documents
    all_tables = []
    for doc in st.session_state.documents:
        for table_info in doc['document_data']['tables']:
            all_tables.append({
                'document': doc['document_data']['filename'],
                'page': table_info['page'],
                'table': table_info['table']
            })
    
    if not all_tables:
        st.info("No tables found in the uploaded documents.")
        return
    
    st.write(f"Found {len(all_tables)} tables across all documents.")
    
    # Display tables and charts
    for i, table_info in enumerate(all_tables):
        with st.expander(f"Table {i+1} - {table_info['document']} (Page {table_info['page']})"):
            table_data = table_info['table']
            
            if not table_data:
                st.warning("Empty table data")
                continue
            
            # Display raw table
            st.subheader("📋 Raw Data")
            try:
                df = pd.DataFrame(table_data[1:], columns=table_data[0])
                st.dataframe(df)
            except Exception as e:
                st.error(f"Could not display table: {str(e)}")
                continue
            
            # Generate chart
            st.subheader("📈 Visualization")
            chart = ChartGenerator.create_chart_from_table(
                table_data, 
                f"Data from {table_info['document']} (Page {table_info['page']})"
            )
            
            if chart:
                # Use unique key for each chart to avoid duplicate ID error
                chart_key = f"chart_{i}_{table_info['document']}_{table_info['page']}"
                st.plotly_chart(chart, use_container_width=True, key=chart_key)
                
                
                # Chart explanation
                explanation = ChartGenerator.explain_chart(table_data)
                st.write("**Chart Explanation:**")
                st.write(explanation)
            else:
                st.info("Could not generate chart from this table data.")

def summaries_interface():
    """Interface for document summaries"""
    st.header("📝 Document Summaries")
    
    if not st.session_state.documents:
        st.info("No documents to summarize.")
        return
    
    # Generate summaries for each document
    for doc in st.session_state.documents:
        with st.expander(f"📄 {doc['document_data']['filename']}"):
            if st.button(f"Generate Summary", key=f"summary_{doc['document_data']['filename']}"):
                if st.session_state.gemma_llm:  # Changed from gemini_llm to gemma_llm
                    with st.spinner("Generating summary..."):
                        summary = st.session_state.gemma_llm.summarize_document(  # Changed from gemini_llm
                            doc['chunks'], 
                            doc['document_data']['metadata']
                        )
                        st.write(summary)
                else:
                    st.error("Please connect to Gemma model first.")  # Updated error message
            
            # Document metadata
            st.subheader("📋 Document Information")
            metadata = doc['document_data']['metadata']
            
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Pages:** {metadata['total_pages']}")
                st.write(f"**Chunks:** {len(doc['chunks'])}")
            
            with col2:
                if metadata.get('title'):
                    st.write(f"**Title:** {metadata['title']}")
                if metadata.get('author'):
                    st.write(f"**Author:** {metadata['author']}")

# CHANGE 8: Update keywords_interface function
def keywords_interface():
    """Interface for keyword extraction"""
    st.header("🔍 Keywords & Topics")
    
    if not st.session_state.documents:
        st.info("No documents to analyze.")
        return
    
    # Extract keywords from all documents
    for doc in st.session_state.documents:
        with st.expander(f"📄 {doc['document_data']['filename']}"):
            if st.button(f"Extract Keywords", key=f"keywords_{doc['document_data']['filename']}"):
                with st.spinner("Extracting keywords..."):
                    # Combine all chunks for keyword extraction
                    full_text = " ".join(doc['chunks'])
                    keywords = st.session_state.gemma_llm.extract_keywords(full_text)  # Changed from gemini_llm
                    
                    # Display keywords as tags
                    st.subheader("🏷️ Keywords")
                    keyword_cols = st.columns(4)
                    for i, keyword in enumerate(keywords):
                        with keyword_cols[i % 4]:
                            st.write(f"🔸 {keyword}")
                    
                    # Word frequency chart
                    if keywords:
                        st.subheader("📊 Keyword Frequency")
                        # Simple frequency analysis
                        words = full_text.lower().split()
                        word_freq = Counter([word for word in words if word in keywords])
                        
                        if word_freq:
                            freq_df = pd.DataFrame(
                                list(word_freq.items()), 
                                columns=['Keyword', 'Frequency']
                            )
                            fig = px.bar(freq_df, x='Keyword', y='Frequency', 
                                       title="Top Keywords by Frequency")
                            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()