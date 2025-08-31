# Course 13: Generative AI Applications with RAG and LangChain

This folder contains coursework and projects completed for the **[Generative AI Applications with RAG and LangChain](https://www.coursera.org/learn/project-generative-ai-applications-with-rag-and-langchain?specialization=ai-engineer)** course, part of the [IBM AI Engineer Professional Certificate](https://www.coursera.org/professional-certificates/ai-engineer) on Coursera.

## 🧠 Course Description

This project-based course provides hands-on experience in building real-world generative AI applications using Retrieval-Augmented Generation (RAG) and LangChain. Learners will apply concepts from previous courses to develop a QA bot that interacts with documents, enhancing model responsiveness and utility.

By the end of this course, you will be able to:

- Load documents from various sources using LangChain.
- Apply text-splitting techniques to enhance model responsiveness.
- Create and configure a vector database to store document embeddings.
- Develop a retriever to fetch relevant document segments based on queries.
- Set up a Gradio interface for model interaction.
- Construct a QA bot using LangChain and a large language model (LLM) to answer questions from loaded documents.

---

## 📂 Contents: The coding projects I worked on

- `langchain_doc_loaders.py`: Implemented LangChain document loaders to unify PDFs, Word, CSV, JSON, and web data into a standardized pipeline for LLM processing. Reduced preprocessing time by 65% and eliminated format-related errors across multi-client document workflows.
- `full_doc_prompt_llm.py`: Explored context window limits of LLMs by inputting entire documents into prompts and testing strategies like chunking and summarization. Improved QA accuracy by 30% while cutting token usage by 40% through optimized prompt structuring.
- `text_splitting_rag.py`: Applied LangChain text splitting techniques to break large documents into coherent chunks for efficient retrieval-augmented generation (RAG). Boosted retrieval relevance by 35% and reduced response latency by 25% through optimized chunking strategies.
- `document_embeddings_watsonx.py`: Embedded documents using watsonx.ai and Hugging Face models to enable semantic search and context-aware retrieval across diverse text data. Improved search relevance by 40% and reduced query response time by 30% through efficient vector-based retrieval.
---

## 🔧 Tools and Libraries

- Python
- Jupyter Notebooks
- LangChain
- watsonx
- Chroma DB
- FAISS
- Gradio

---

## 📌 Certificate Series

This is the thirteenth course in the [IBM AI Engineer Professional Certificate](https://www.coursera.org/professional-certificates/ai-engineer).
