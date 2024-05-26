# Social-Media-Analytics

## 專案說明：
+ **對 ptt 股市版面關鍵字為「台積電」的文章進行文本分析、情緒分析** <br>
+ **將新聞網的版別的文章組合起來，使用 Decision Tree、Logistic Regression、SVM、Random Forest 等方式訓練模型，使用 Cross-Validation 、F1-Score 評估模型的性能，選出 best model，使模型能夠預測新聞版別，再用分類模型進行文件的版別分類** <br>
+ **利用主題模型，從大量文本數據中提取和理解隱藏的主題** <br>
+ **訓練、使用 Word2Vec 模型，使用 Bert、LLAMA 2、Cohere 等模型，透過 sentence-transformers 以及 API 取得 embeddings** <br>
+ **使用 BERT 相關資源（例如網路上別人訓練好的 BERT 模型）來達成 Token classification、Sequence classification 以及 Text Clustering 等等相關文字處理任務** <br>
+ **LLM Information Extraction，使用 Langchian 套件與 HuggingFace 模型來實作LLM Inference、Prompt Engineering、Information Extraction(Sentiment classification, NER）** <br>
+ **使用 RAG 結合 0050 簡式公開說明書，能夠回答使用者的提問** <br>

## 套件說明：
- Jieba : 中文斷詞套件
- wordcloud : 文字雲繪圖工具
- CKIP : 全名為 Chinese Knowledge and Information Processing，中研院開發的中文自然語言處理工具
- SnowNLP : SnowNLP是一個可以方便的處理中文文本內容的python類庫，主要功能包括斷詞、詞性標註、情緒分析、漢字轉拼音、繁體轉簡體、關鍵詞提取以及文本摘要等等
- NLTK : 全名為Natural Language Tool Kit，自然語言處理工具
- CountVectorizer, TfidfTransformer : sklearn 中計算詞頻與 tf-idf 的套件
- networkx : 網絡圖建構、繪製工具
- numpy.linalg : 矩陣與向量運算套件
- gensim : 主題模型函式庫
- pyLDAvis : 互動式LDA視覺化套件
- LCEL (LangChain Expression Language) : LCEL 可以讓我們更簡單的開發複雜的 chain 應用，可以支援：Streaming, Parallelism, logging。

## 文字處理基本流程
+ **資料初步清理：** <br>
將文字內容轉為正規的語句，例如：去除特定標籤、符號、統一標點符號的使用等等。<br><br>
+ **斷句斷詞：** <br>
使用工具區隔文章中不同的句子、詞彙 <br><br>
+ **去除停用字：** <br>
停用字就是與分析無關的詞彙，甚至這些詞彙可能會影響分析的結果。   
因此我們必須在資料處理的過程中將其排除。例如：語助詞 <br><br>
+ **產生結構化資料：** <br>
根據需求產生不同結構化資料(Tidy Data)，以供後續分析使用。 <br><br>
+ **應用更多分析：** <br>
進行不同的分析，例如：詞頻分析、情緒分析、文辭和文件分析、文件分類、社會網路分析等等。

## 情緒分析
+ **Lexicon-based 情緒分析：** <br>
進行基於情緒字典的情緒分析，首先需要準備情緒字典後再將字典與斷詞後的文章進行合併<br><br>
+ **SnowNLP Courpus-base 情緒分析：** <br>
SnowNLP是Courpus-base的情緒分析，不是使用情緒字典，而是使用預訓練的情緒分析模型或演算法，給與整個文章一個情緒分數。<br><br>

## 文本分析
+ **以 TFIDF 找出文本內找出重要詞彙** <br>
+ **透過 Jieba 斷詞與 N-gram 幫助建立斷詞字典** <br>
+ **以 Pearson correlation 計算兩個詞彙間的相關性** <br>
+ **以 建立 Ngram 預測模型** <br>

## 文件分類
+ **使用文章轉 DTM(document term matrix)的方式，將文章用不同的字詞分布表示，再利用 sklearn 套件，套用 Decision Tree、Logistic Regression、SVM、Random Forest 等訓練模型，分辨不同的文件** <br>
+ **使用 Cross-Validation 、F1-Score 評估模型的性能，選出 best model** <br>

## 主題模型
+ **文件主題模型採用非監督式學習的方式，不事先人為做標籤與註解，訓練的資料是從資料文本得來，讓我們可以規模化給予大量文本，而訓練出主題模型，其中最為知名為LDA模型，用於從大量文本數據中提取和理解隱藏的主題** <br>
+ **LDA 模型指標** <br>
Pointwise Mutual Information (PMI) : 自然語言處理中，想要探討兩個字之間是否存在某種關係。例如：某些字會一起出現，可能帶有某些訊息，因此這個可以用 PMI 來計算，數字越大越好。<br>
perplexity : perplexity 也是評估的指標之一，廣泛用於語言模型的評估，意思為複雜度，因此數字要越小越好。

## Text Representation
+ **Embeddings** <br>
在自然語言處理（NLP）和機器學習領域中指的是將高維的稀疏數據（如詞語、句子、文檔等）轉換為低維的密集向量的技術。這些低維向量可以捕捉原始數據的語義信息和結構特徵，使得機器學習模型能夠更有效地處理和理解這些數據。
+ **Encoder & Decoder** <br>
Encoder 將輸入序列轉換為一個固定長度的上下文向量（context vector），該向量捕捉輸入序列的語義訊息<br>
Decoder 根據 Encoder 生成的上下文向量，生成輸出序列。<br>
+ **Word2Vec 模型** <br>
將詞語轉換為向量，並捕捉詞語之間的語義關係
+ **Word2Vec 原理** <br>
CBOW（Continuous Bag of Words）：使用上下文詞語來預測中心詞。例如，給定上下文詞語 "The sat on the floor" 來預測中心詞 "cat"。 <br>
Skip-gram：與 CBOW 相反，使用中心詞來預測上下文詞語。例如，給定中心詞 "cat" 來預測上下文詞語 "The", "sat", "on", "the", "floor"。 <br>
訓練 Word2Vec 模型、使用別人訓練好的 Word2Vec 模型
+ **Transformers Embeddings** <br>
小模型（BERT）：不同語言的BERT：uncased / chinese / multilingual <br>
大模型（LLM）：API based、Open Source LLM <br>
+ **Transformers Embeddings 特點** <br>
Self-Attention：自注意力機制允許模型在計算每個詞的嵌入時，關注序列中的其他詞，從而捕捉長距離依賴關係。<br>
Multi-Head Attention：能夠同時捕捉不同子空間中的訊息，增強模型的表示能力。<br>
+ **資料集實作任務** <br>
使用 embedding 做 NLP 任務、找相似文件（文章）、文件分類任務 <br>
找相似文件（文章)：計算目標文章的 embedding 與所有文章的 embedding 之間的 Cosine Similarity <br>
文件分類任務：使用 bert-base-chinese 模型對新聞資料集做 embeddings，接著訓練分類器

## BERT (Encoder-only-model)
BERT 相關下游任務的程式碼應用，使用 BERT 相關資源（例如網路上別人訓練好的 BERT 模型）來達成 Token classification、Sequence classification 以及 Text Clustering 等等相關文字處理任務
+ **雙向性（Bidirectionality）** <br>
在傳統的單向模型中，模型只能從左到右（或從右到左）進行預測，而 BERT 可以同時考慮到兩邊的上下文訊息，從而更好地捕捉詞語的語義訊息。<br>
+ **預訓練和微調（Pre-training and Fine-tuning）** <br>
BERT 首先在大規模文本語料上進行預訓練，然後在特定任務上進行微調。預訓練過程包括兩個任務，遮罩語言模型（MLM）：在預訓練過程中，BERT 隨機遮罩掉輸入文本中的一些單詞，然後要求模型預測這些被遮罩的單詞ㄤ;下一句預測（NSP）：BERT 接收成對的句子，並預測第二個句子是否是第一個句子的下一句。<br>
+ **多任務學習（Multi-task Learning）** <br>
BERT 可以應用於多種 NLP 任務，如文本分類、命名實體識別（NER）、問答系統、文本生成等。通過微調，BERT 在這些任務中都能取得優異的效果。

## LLM Information Extraction
使用 Langchian 套件與 HuggingFace 模型來實作
+ **LLM Inference** <br>
+ **Prompt Engineering** <br>
Few-shot、Chain of thought(CoT)、Self-consistency
+ **Information Extraction(Sentiment classification, NER)** <br>
+ **RAG (Retrieval-Augmented Generation) 檢索增強生成** <br>
RAG 模型的基本原理是將檢索和生成兩個步驟結合起來。RAG 模型使用檢索模型從外部知識庫中檢索與用戶查詢相關的資訊。然後，RAG 模型使用生成模塊對檢索到的資訊進行處理，生成最終的答案。
Prompt 的設計有很大程度會影響 LLM 的輸出，甚至影響 LLM 整體能力，以下會介紹幾個常用的 prompt 技巧






