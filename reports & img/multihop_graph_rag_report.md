

# Week 6 Hands-On Report

## Introduction
In this week’s hands-on assignment, I explored advanced retrieval-augmented generation techniques, focusing on Graph-RAG and Multi-Hop QA. These extend the work from Week 4 (baseline dense retrieval) and Week 5 (advanced RAG with rerankers and improved chunking). The goal was to evaluate whether graph-based retrieval and multi-hop reasoning provide measurable improvements for answering complex queries over a domain-specific corpus.  

This report connects the results from my experiments to the objectives of my project, which requires building reliable, explainable, and scalable retrieval systems for complex question answering.

## Key Results
I implemented two tracks:  
- **Track A: Graph-RAG** — retrieval grounded in entity–relation neighborhoods.  
- **Track B: Multi-Hop QA** — question decomposition and chained reasoning.  

The ablation study compared these against the Week 4 baseline using accuracy, faithfulness, and guardrail metrics:

- **Baseline Accuracy:** ~0.60  
- **Graph-RAG Accuracy:** ~0.69  
- **Multi-Hop RAG Accuracy:** ~0.68  

This shows that Graph-RAG improved correctness by nearly 9 percentage points over the baseline, and Multi-Hop RAG also provided a measurable improvement.  

Faithfulness scores revealed an interesting trade-off:  
- **Baseline Faithfulness:** ~0.85 (highest)  
- **Graph-RAG Faithfulness:** ~0.78  
- **Multi-Hop RAG Faithfulness:** ~0.78  

While Graph-RAG and Multi-Hop produced more correct answers overall, they sometimes introduced extra reasoning noise compared to the baseline.  

Guardrail performance was broadly similar across models, showing that safety filters were consistent regardless of retrieval style.  

Overall, the evaluation confirmed that structured retrieval and reasoning improved accuracy, but with a slight reduction in strict factual grounding.

## Connection to My Project
My project involves building a domain-adapted RAG pipeline where questions are rarely simple one-hop lookups. Queries often involve multiple entities (e.g., “Which team proposed Method X, and what datasets validated it?”) that demand compositional reasoning.  

- **Graph-RAG Contribution:** For my project, the ~9% improvement in accuracy over the baseline is highly significant. The graph-based retriever provides a principled way to ground answers in explicit evidence, connecting datasets, methods, and results through entity relations. This is crucial for my corpus, where relationships drive correctness.  
- **Multi-Hop QA Contribution:** Multi-hop reasoning provided ~8% better accuracy than the baseline, demonstrating its usefulness for compositional questions. Although faithfulness scores were slightly lower, the ability to handle multi-entity, multi-step queries aligns directly with my project’s needs.  

These results show that both techniques are directly applicable to my project’s goal of answering complex, evidence-based questions.

## Challenges and Lessons Learned
Several challenges were encountered that mirror the needs of my project:  

- **Environment Setup:** Dependency conflicts when integrating LangChain, sentence-transformers, rerankers, and LLM backends had to be resolved carefully. The environment file (`multihop_graph_rag_env.json`) ensured reproducibility.  
- **Run Configurations:** Setting retriever (`intfloat/e5-large`), reranker (MiniLM), generator (TinyLlama), hop limits, and graph weighting required careful tuning, as documented in `multihop_graph_rag_run_config.json`.  
- **Graph Construction:** Extracting clean entities and relations required balancing between sparse and overly dense graphs. Sparse graphs underperformed, while dense graphs reduced efficiency.  
- **Multi-Hop Stability:** Chained reasoning was sensitive to errors in intermediate hops, which explains why accuracy improved but faithfulness dropped. This emphasized the need for more robust decomposition strategies.  

These lessons directly inform my project design, particularly in preprocessing pipelines and handling error propagation in multi-hop reasoning.

## Conclusion
The Week 6 hands-on demonstrated that Graph-RAG and Multi-Hop QA can significantly improve retrieval accuracy over a baseline dense retriever, with Graph-RAG reaching ~0.69 accuracy and Multi-Hop ~0.68, compared to ~0.60 for the baseline. However, both methods showed slightly lower faithfulness than the baseline (0.78 vs. 0.85), indicating a trade-off between correctness and strict grounding.  

For my project, these methods are directly applicable: Graph-RAG enables grounded and interpretable retrieval, while Multi-Hop QA provides the reasoning capability necessary for compositional queries. The insights gained from the ablation study, particularly the accuracy–faithfulness trade-off, will guide how I integrate advanced retrieval techniques into a robust end-to-end system.

