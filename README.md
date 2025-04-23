# Social-Media-Analytics

## Project Description

- Analyze PTT Stock posts mentioning “TSMC” and aggregate news articles by section.
- Train text classifiers (Decision Tree, SVM, Logistic Regression, Random Forest) with cross-validation and F1-score evaluation for news categorization.
- Extract and visualize latent topics using LDA and pyLDAvis.
- Generate embeddings (Word2Vec, BERT, LLAMA 2) via Sentence-Transformers or APIs for classification and similarity searches.
- Implement LLM-based information extraction (sentiment analysis, NER) with LangChain & HuggingFace and build a RAG Q&A system using prospectus data.
- Visualize social and entity networks using NetworkX and PyVis.

## Dependencies

- **scikit-learn**: Modeling, cross-validation, vectorization
- **Gensim & pyLDAvis**: LDA topic modeling and visualization, Word2Vec
- **transformers & sentence-transformers**: BERT and LLAMA 2 embeddings
- **LangChain (LCEL)**: Streamlined LLM inference pipelines
- **NetworkX & PyVis**: Network graph construction and visualization

## Pipeline & Methods

1. **Preprocessing**: Clean text, tokenize, remove stopwords.
2. **Feature Extraction**: TF-IDF, N-grams, dense embeddings.
3. **Modeling**:
   - **Classification**: Train and select best model via CV & F1.
   - **Topic Modeling**: LDA with PMI and perplexity metrics.
4. **LLM Techniques**: Fine-tune BERT for sequence/token classification; apply prompt engineering (few-shot, CoT).
5. **RAG System**: Retrieve prospectus passages and generate answers with a generative model.

## Visualization

- **pyLDAvis**: Interactive topic exploration.
- **NetworkX & PyVis**: Social and entity relationship graphs.
