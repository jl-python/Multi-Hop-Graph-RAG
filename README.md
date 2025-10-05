# Multi-Hop-Graph-RAG
This project dives into a graph knowledge base for information retrieval and examines the performances between a BM25+RRF+Reranker, Graph, and Multi Hop Graph RAG systems in answering queries based on text inputs.


# Multi-Hop Graph RAG (Streamlit App)
This repository contains a **Streamlit application** for running a Multi-Hop Retrieval-Augmented Generation (RAG) pipeline using graph-based reasoning.

The app will allow uploading documents where it splits multi part questions into a smaller parts and answers through the LLM model pulling info from the knowledge graph implemented into this RAG system. 

You can view the graph and top k retrieval texts after submitting the your question on the app. 


#### - Install Requirements & Dependencies

    Run these in a terminal 

    pip install streamlit langchain sentence-transformers transformers chromadb spacy pypdf rank-bm25 accelerate
    python -m spacy download en_core_web_sm 


#### - Running File
    Run this in terminal
    streamlit run rag/week_6_streamlit.py
    

#### - App Features

* Retrieval Controls
    
      Top-K: Choose how many documents are fetched per query (3–10). Default = 5.
    
      Max Hops: Set how many multi-hop steps are allowed (1–4).
    
      LLM Temperature: Select the verbosity and creativity of the LLM (0.20-1.00)

* LLM Type

      TinyLlama/TinyLlama-1.1B-Chat-v1.0
      
      gpt2 (baseline)


* Cross-Encoder Reranker (optional)
  
      Improves retrieval accuracy using sentence-transformers CrossEncoder, but is slower on CPU.


* SpaCy Transformer Toggle (optional)

      Use SpaCy’s transformer pipeline for entity extraction (GPU recommended).


