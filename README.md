# Social-Media-Analytics

## Project Description

- Perform text processing and sentiment analysis on PTT Stock board posts containing the keyword "TSMC".
- Aggregate news articles by section and train classification models (Decision Tree, Logistic Regression, SVM, Random Forest) to predict article categories.
- Apply topic modeling (LDA) to extract and interpret hidden themes from large text corpora.
- Train and utilize Word2Vec embeddings; obtain sentence embeddings via BERT, LLAMA 2, and Cohere using Sentence-Transformers and respective APIs.
- Leverage pre-trained BERT models for token classification, sequence classification, and text clustering tasks.
- Implement LLM-based information extraction using LangChain and HuggingFace (sentiment classification, NER).
- Develop a Retrieval-Augmented Generation (RAG) Q&A system integrating the 0050 simplified prospectus.
- Visualize social graphs and entity relationships with PyVis and NetworkX.

## Dependencies

- **scikit-learn**: Model training, evaluation (Decision Tree, SVM, Logistic Regression, Random Forest), vectorization (CountVectorizer, TfidfTransformer)
- **Gensim**: Topic modeling (LDA) and Word2Vec embedding training
- **pyLDAvis**: Interactive visualization for topic models
- **LangChain (LCEL)**: Develop complex LLM inference chains, including streaming and parallelism
- **HuggingFace Transformers**: Utilize pre-trained BERT and LLAMA 2 models for embeddings, sequence and token classification
- **Sentence-Transformers**: Generate high-quality sentence embeddings for semantic analysis
- **NetworkX & PyVis**: Construct and visualize social graphs and entity relationship networks

## Text Processing Pipeline

1. **Data Cleaning**: Normalize text, remove unwanted tags/symbols, unify punctuation.
2. **Sentence Segmentation & Tokenization**: Split text into sentences and tokens.
3. **Stopword Removal**: Remove irrelevant words that may bias analysis.
4. **Structured Data Generation**: Convert cleaned text into tidy, structured formats for analysis.
5. **Analytical Applications**: Conduct tasks such as frequency analysis, sentiment analysis, document classification, and social network analysis.

## Sentiment Analysis

- **Lexicon-Based**: Combine tokenized text with a sentiment lexicon to score emotions.
- **SnowNLP Corpus-Based**: Use SnowNLP’s pre-trained sentiment model to assign sentiment scores at the document level.

## Text Analysis

- **TF-IDF**: Identify important terms within documents.
- **N-Gram Dictionary**: Build token dictionaries using Jieba and N-grams.
- **Word Correlation**: Calculate Pearson correlation between word pairs.
- **N-Gram Prediction**: Develop predictive models based on N-gram features.

## Document Classification

- Represent articles as a Document-Term Matrix (DTM).
- Train classifiers (Decision Tree, Logistic Regression, SVM, Random Forest).
- Evaluate using Cross-Validation and F1-score to select the best-performing model.

## Topic Modeling

- Apply unsupervised LDA to discover latent topics without manual labeling.
- Evaluate model quality using:
  - **Pointwise Mutual Information (PMI)**: Measures word association strength (higher is better).
  - **Perplexity**: Assesses model complexity (lower is better).

## Text Representation

- **Embeddings**: Map sparse textual data to dense, low-dimensional vectors capturing semantic information.
- **Encoder-Decoder Architecture**: Convert input sequences into context vectors and generate output sequences.
- **Word2Vec**: CBOW and Skip-gram models for capturing word relationships.
- **Transformer Embeddings**:
  - **BERT**: Base and multilingual variants for contextual embeddings.
  - **LLMs**: API-based or open-source large language models.
- **Applications**: Similarity search (Cosine similarity between embeddings), document classification.

## BERT Applications

- Fine-tune pre-trained BERT models for downstream tasks: token classification, sequence classification, text clustering.
- Leverage BERT’s bidirectional context, Masked Language Modeling (MLM), and Next Sentence Prediction (NSP) objectives.

## LLM Information Extraction

- **LLM Inference**: Run predictions with large language models.
- **Prompt Engineering**: Few-shot examples, Chain-of-Thought (CoT), self-consistency.
- **Information Extraction**: Sentiment classification, Named Entity Recognition (NER).
- **RAG (Retrieval-Augmented Generation)**: Combine retrieval from a knowledge base with generative models to produce grounded answers.
