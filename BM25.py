from collections import defaultdict
from pathlib import Path
import pickle
from Reader import MultiFileReader
from contextlib import closing
import math

TUPLE_SIZE = 6
TF_MASK = 2 ** 16 - 1  # Masking the 16 low bits of an integer

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

def read_index(base_dir, name):
    with open(Path(base_dir) / f'{name}.pkl', 'rb') as f:
        return pickle.load(f)
    
body_idx = read_index('./body_index/', 'body_index')
doc_len = body_idx.doc_len_mapping

class BM25:
    """
    Best Match 25.

    Parameters to tune
    ----------
    k1 : float, default 1.5

    b : float, default 0.75

    Attributes
    ----------
    tf_ : list[dict[str, int]]
        Term Frequency per document. So [{'hi': 1}] means
        the first document contains the term 'hi' 1 time.
        The frequnecy is normilzied by the max term frequency for each document.

    doc_len_ : list[int]
        Number of terms per document. So [3] means the first
        document contains 3 terms.

    df_ : dict[str, int]
        Document Frequency per term. i.e. Number of documents in the
        corpus that contains the term.

    avg_doc_len_ : float
        Average number of terms for documents in the corpus.

    idf_ : dict[str, float]
        Inverse Document Frequency per term.
    """

    def __init__(self, inverted, path, query, k1=1.5, b=0.75 ):

        self.path = path
        self.query = query
        self.inverted = inverted
        self.b = b
        self.k1 = k1
        self.N_ = len(doc_len)
        self.avgdl_ = sum(doc_len) / len(doc_len)
        self.candidates_tf = defaultdict(list)
        self.idf_dic = defaultdict(list)

    def calc_idf(self):
        """
        This function calculate the idf values according to the BM25 idf formula for each term in the query.

        Parameters:
        -----------
        query: list of token representing the query. For example: ['look', 'blue', 'sky']

        Returns:
        -----------
        idf: dictionary of idf scores. As follows:
                                                    key: term
                                                    value: bm25 idf score
        """
        # YOUR CODE HERE
        idf_dic = {}

        for word in self.query:
            # if the term already added
            if word in idf_dic:
                continue

            else:
                # if the term exist in the corpus
                words_posting_lst = read_posting_list(self.inverted, self.path, word)
                if len(words_posting_lst) > 0:
                    idf_dic[word] = math.log10(
                        ((self.N_ - len(words_posting_lst) + 0.5) / (len(words_posting_lst) + 0.5)) + 1)

        self.idf_dic = idf_dic

    def doc_score(self, doc_id):
        """
        This function calculate the bm25 score for given query and document.

        Parameters:
        -----------
        query: list of token representing the query. For example: ['look', 'blue', 'sky']
        doc_id: integer, document id.

        Returns:
        -----------
        score: float, bm25 score.
        """
        # YOUR CODE HERE
        score = 0

        for word in self.query:
            curr_tf = ""
            if doc_id in self.candidates_tf:
                for tup in self.candidates_tf[doc_id]:
                    if tup[0] == word:
                        curr_tf = tup[1]
                        continue
            if word in self.idf_dic and doc_id in doc_len:
                if curr_tf != "":
                    score += ((self.idf_dic[word]) * curr_tf * (self.k1 + 1)) / (
                            curr_tf + self.k1 * (
                            1 - self.b + ((self.b * doc_len[doc_id]) / self.avgdl_)))
        return score

    def create_tf(self):
        c = {}
        for word in self.query:
            words_posting_lst = read_posting_list(self.inverted, self.path, word)
            for tup in words_posting_lst:
                    c[tup[0]] = []
                    c[tup[0]].append((word, tup[1]))
        self.candidates_tf = c

    def calc_candidates_score(self):
        scores = {}
        self.create_tf()
        self.calc_idf()
        for c in self.candidates_tf:
            scores[c] = self.doc_score(c)
        return scores
