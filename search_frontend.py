import math
from collections import Counter
from contextlib import closing

import nltk
import pandas as pd
from flask import Flask, request, jsonify

from BM25 import BM25
from inverted_index_anchor_colab import InvertedIndex as anchorINV
from inverted_index_body_colab import InvertedIndex as bodyINV
from inverted_index_title_colab import InvertedIndex as titleINV
from Reader import MultiFileReader

nltk.download('stopwords')

from nltk.stem.porter import *
from nltk.corpus import stopwords

TUPLE_SIZE = 6
TF_MASK = 2 ** 16 - 1

# Paths
body_path = './body_index/'
title_path = './title_index/'
anchor_path = './anchor_index/'
page_rank_path = './pagerank/'
page_view_path = './pageview'

# Create inverted index instance
inverted_index_body = bodyINV.read_index(body_path, 'body_index')
inverted_index_title = titleINV.read_index(title_path, 'title_index')
inverted_index_anchor = anchorINV.read_index(anchor_path, 'anchor_index')
page_rank = pd.read_csv(page_rank_path + 'pagerank.csv', index_col=0, header=None, squeeze=True)
page_view = bodyINV.read_index(page_view_path, 'pageview')


class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)

app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

@app.route("/")
def home_page():
    return jsonify("Welcome to Wikipedia Search Engine! creators : Eden Chai, Tslil Brami ")

@app.route("/search")
def search():
    """ Returns up to a 100 of your best search results for the query. This is
        the place to put forward your best search engine, and you are free to
        implement the retrieval whoever you'd like within the bound of the
        project requirements (efficiency, quality, etc.). That means it is up to
        you to decide on whether to use stemming, remove stopwords, use
        PageRank, query expansion, etc.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each
        element is a tuple (wiki_id, title).
    """
    # calc using BM25
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    
    # BEGIN SOLUTION
    tokenized_query = token_query(query)
    if len(query) == 0:
        return jsonify(res)
    
    body_bm25 = BM25(inverted_index_body, body_path, tokenized_query)
    body_scores = body_bm25.calc_candidates_score()

    title_bm25 = BM25(inverted_index_title, title_path, tokenized_query)
    title_scores = title_bm25.calc_candidates_score()
    
    anchor_bm25 = BM25(inverted_index_anchor, anchor_path, tokenized_query)
    anchor_scores = anchor_bm25.calc_candidates_score()
    
    final_scores = merge_results(body_scores, title_scores, anchor_scores)
    
    i = 0
    for doc_id in final_scores:
        if i >= 100:
            break
        if doc_id in inverted_index_body.doc_title_mapping:
            res.append((doc_id, inverted_index_body.doc_title_mapping[doc_id]))
        i += 1

    return jsonify(res)
    
@app.route("/search_body")
def search_body():
    """ Returns up to a 100 search results for the query using TFIDF AND COSINE
        SIMILARITY OF THE BODY OF ARTICLES ONLY. DO NOT use stemming. DO USE the
        staff-provided tokenizer from Assignment 3 (GCP part) to do the
        tokenization and remove stopwords.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search_body?query=political+hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each
        element is a tuple (wiki_id, title).
    """
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)

    tokenized_query = token_query(query)
    if len(tokenized_query) == 0:
        return jsonify(res)

    # BEGIN SOLUTION
    word_weight_in_query = { }
    cosinesim = { }
    word_weight_in_q = Counter(tokenized_query)
    for word in word_weight_in_q:
        word_weight_in_query[word] = word_weight_in_q[word] / len(tokenized_query)

    # calc cosine similarity by the algorithm on lecture 3
    # upper product calculation
    product_for_bottom_from_query = 0
    for word in word_weight_in_q:
        weight = word_weight_in_query[word]
        product_for_bottom_from_query += math.pow(weight, 2)
        if word not in inverted_index_body.idf:
            continue
        idf = inverted_index_body.idf[word]
        word_posting_lst = read_posting_list(inverted_index_body, body_path, word)
        if len(word_posting_lst) > 0:
            for tup in word_posting_lst:
                if tup[0] not in cosinesim:
                    cosinesim[tup[0]] = (tup[1] / inverted_index_body.doc_len_mapping[tup[0]]) * idf * weight
                else:
                    cosinesim[tup[0]] += (tup[1] / inverted_index_body.doc_len_mapping[tup[0]]) * idf * weight

    if len(cosinesim) < 1:
        return jsonify(res)
    
    # calc the bottom product for cosinesim
    for doc_id in cosinesim:
        dominator = math.sqrt(inverted_index_body.dominator_mapping[doc_id] * product_for_bottom_from_query)
        cosinesim[doc_id] = cosinesim[doc_id] / dominator

    sorted_cosinesim = { k: v for k, v in sorted(cosinesim.items(), key=lambda item: item[1], reverse=True) }
    i = 0
    for doc in sorted_cosinesim:
        if i >= 100:
            break
        res.append((doc, inverted_index_body.doc_title_mapping[doc]))
        i += 1

    return jsonify(res)

@app.route("/search_title")
def search_title():
    """ Returns ALL (not just top 100) search results that contain A QUERY WORD
        IN THE TITLE of articles, ordered in descending order of the NUMBER OF
        QUERY WORDS that appear in the title. DO NOT use stemming. DO USE the
        staff-provided tokenizer from Assignment 3 (GCP part) to do the
        tokenization and remove stopwords. For example, a document with a
        title that matches two of the query words will be ranked before a
        document with a title that matches only one query term.

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_title?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to
        worst where each element is a tuple (wiki_id, title).
    """
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)

    # BEGIN SOLUTION
    tokenized_query = token_query(query)
    if len(tokenized_query) == 0:
        return jsonify(res)

    # collecting docs that query's words appear in
    query_binary_similarity = {}
    for word in tokenized_query:
        posting_lst = read_posting_list(inverted_index_title, title_path, word)
        if len(posting_lst) > 0:
            for doc in posting_lst:
                if doc[0] in query_binary_similarity:
                    query_binary_similarity[doc[0]] += 1
                else:
                    query_binary_similarity[doc[0]] = 1

    if len(query_binary_similarity) == 0:
        return jsonify(res)

    sorted_query_similarity = {k: v for k, v in sorted(query_binary_similarity.items(), key=lambda x: x[1], reverse=True)}
    for key in sorted_query_similarity:
        if key in inverted_index_title.doc_title_mapping:
            res.append((key, inverted_index_title.doc_title_mapping[key]))

    # END SOLUTION
    return jsonify(res)


@app.route("/search_anchor")
def search_anchor():
    """ Returns ALL (not just top 100) search results that contain A QUERY WORD
        IN THE ANCHOR TEXT of articles, ordered in descending order of the
        NUMBER OF QUERY WORDS that appear in anchor text linking to the page.
        DO NOT use stemming. DO USE the staff-provided tokenizer from
        Assignment 3 (GCP part) to do the tokenization and remove stopwords.
        For example, a document with a anchor text that matches two of the
        query words will be ranked before a document with anchor text that
        matches only one query term.

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_anchor?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to
        worst where each element is a tuple (wiki_id, title).
    """
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    
    # BEGIN SOLUTION
    tokenized_query = token_query(query)
    if len(tokenized_query) == 0:
        return jsonify(res)
    
    query_binary_similarity = {}
    for word in tokenized_query:
        posting_lst = read_posting_list(inverted_index_anchor, anchor_path, word)
        if len(posting_lst) > 0:
            for doc in posting_lst:
                if doc[0] in query_binary_similarity:
                    query_binary_similarity[doc[0]] += 1
                else:
                    query_binary_similarity[doc[0]] = 1

    if len(query_binary_similarity) == 0:
        return jsonify(res)

    sorted_query_similarity = {k: v for k, v in sorted(query_binary_similarity.items(), key=lambda item: item[1], reverse=True)}
    for key in sorted_query_similarity:
        if key in inverted_index_anchor.doc_title_mapping:
            res.append((key, inverted_index_anchor.doc_title_mapping[key]))
    # END SOLUTION
    return jsonify(res)


@app.route("/get_pagerank", methods=['POST'])
def get_pagerank():
    """ Returns PageRank values for a list of provided wiki article IDs.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pagerank
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pagerank', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of floats:
          list of PageRank scores that correrspond to the provided article IDs.
    """
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
        return jsonify(res)

    # BEGIN SOLUTION
    page_rank_dictionary = page_rank.to_dict()
    notExist = 0
    
    for page_id in wiki_ids:
        if page_id in page_rank_dictionary:
            res.append(page_rank_dictionary[page_id])
        else:
            res.append(notExist)
    # END SOLUTION
    
    return jsonify(res)


@app.route("/get_pageview", methods=['POST'])
def get_pageview():
    """ Returns the number of page views that each of the provide wiki articles
        had in August 2021.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pageview
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pageview', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ints:
          list of page view numbers from August 2021 that correrspond to the
          provided list article IDs.
    """
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
        return jsonify(res)
    
    # BEGIN SOLUTION
    page_view_dictionary = page_view
    notExist = 0
    for page in wiki_ids:
        if page in page_view_dictionary:
            res.append(page_view_dictionary[page])
        else:
            res.append(notExist)
    # END SOLUTION
    
    return jsonify(res)

""" ------------------------------ Auxiliary functions ------------------------------ """

def token_query(query):
    english_stopwords = frozenset(stopwords.words('english'))
    corpus_stopwords = ["category", "references", "also", "external", "links", "may", "first",
                        "see", "history", "people", "one", "two", "part", "thumb", "including",
                        "second", "following", "many", "however", "would", "became"]
    RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)
    all_stopwords = english_stopwords.union(corpus_stopwords)
    tokensQ = [token.group() for token in RE_WORD.finditer(query.lower())]
    filteredQ = [tok for tok in tokensQ if tok not in all_stopwords]
    return filteredQ

def read_posting_list(inverted, path, w):
    with closing(MultiFileReader(path)) as reader:
        locs = inverted.posting_locs[w]
        posting_list = []
        if len(locs) > 0:
            b = reader.read(locs, inverted.df[w] * TUPLE_SIZE)
            for i in range(inverted.df[w]):
                doc_id = int.from_bytes(b[i * TUPLE_SIZE:i * TUPLE_SIZE + 4], 'big')
                tf = int.from_bytes(b[i * TUPLE_SIZE + 4:(i + 1) * TUPLE_SIZE], 'big')
                posting_list.append((doc_id, tf))
        return posting_list

def merge_results(body_scores, title_scores, anchor_scores, body_weight=1/3, title_weight=1/3, anchor_weight=1/3):
    res = {}
    
    # body score
    for doc_id, score in body_scores.items():
        if doc_id in res:
            res[doc_id] += score * body_weight
        else:
            res[doc_id] = score * body_weight
    
    # title score
    for doc_id, score in title_scores.items():
        if doc_id in res:
            res[doc_id] += score * title_weight
        else:
            res[doc_id] = score * title_weight
            
    # anchor score
    for doc_id, score in anchor_scores.items():
        if doc_id in res:
            res[doc_id] += score * anchor_weight
        else:
            res[doc_id] = score * anchor_weight

    sorted_scores = {k: v for k, v in sorted(res.items(), key=lambda item: item[1], reverse=True)}
    return sorted_scores
    

""" ----------------------------------- Main ----------------------------------- """

if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080, debug=False)
