# Wikipedia Search Engine
## Description
The primary goal was, given a query to retrieve the most relevant article from the corpus, obviously at the best runtime as possible.

![Wikipedia](https://user-images.githubusercontent.com/73795171/147779008-506dad5d-9adb-4e28-bb61-a37bdd014c70.jpg)

## Dadaset
* Entire Wikipedia dump in a shared Google Storage bucket.
* Pageviews for articles.
* Queries and a ranked list of up to 100 relevant results for them.

## Preprocess of the data
* **Reader -**  read all Wikipedia articles.
* **Parser -**  parse and clean Wikipedia page's entries. 
* **Tokenizer -**  tokenize words by using nltk word tokenize and stopwords.
* **Indexer -**  build inverted index for body, title and anchor text of the articles.

## Ranking methods
* **Cosine Similarity using TF-IDF -** on the body of articles.
* **Binary ranking using VSM -** on the title and anchor text of articles.
* **BM25 -** calculating the score of each part in the articles and them merge the results.
* **Page Rank**
* **Page Views**

## Evaluation
The quality of results tested and evaluated with MAP@40, Recall, Precision measurements.

## 💡 Platforms
* PyCharm in Python 3.7
* Google Colaboratory
* Google Cloud Platform
* VM in Compute Engine
