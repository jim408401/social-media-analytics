# Social-Media-Analytics

## Project Overview

- Analyze PTT Stock posts mentioning “TSMC” and aggregate news articles by section.
- Train text classifiers (Decision Tree, SVM, Logistic Regression, Random Forest) with cross-validation and F1-score evaluation for news categorization.
- Extract and visualize latent topics using LDA and pyLDAvis.
- Generate embeddings (Word2Vec, BERT, LLAMA 2) via Sentence-Transformers or APIs for classification and similarity searches.
- Implement LLM-based information extraction (sentiment analysis, NER) with LangChain & HuggingFace and build a RAG Q&A system using prospectus data.
- Visualize social and entity networks using NetworkX and PyVis.

## Pipeline & Methods

1. **Text Preprocessing & Cleaning**: Normalize text, tokenize, and remove stopwords.
2. **Feature Extraction**: Build TF-IDF vectors, N-grams, and dense embeddings.
3. **Model Training & Evaluation**: Train Decision Tree, SVM, Logistic Regression, Random Forest and select the best model via cross-validation and F1-score.  
4. **LLM Integration**: Fine-tune BERT for token/sequence classification and employ prompt engineering (few‑shot, CoT) for information extraction tasks (sentiment, NER).
5. **RAG Q&A System**: Combine retrieval of prospectus passages with a generative model to deliver accurate answers.
