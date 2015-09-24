#!/usr/bin/env pytho.
"""
This is a naive implementation of Sparse LDA:
    http://people.cs.umass.edu/~lmyao/papers/fast-topic-model10.pdf

LDA is a topic modeling algorithm developed in the early 2000's.  Given a
corpus of documents and an integer N it outputs the N multinomial distributions
over words ('topics') it thinks were used to generate the corpus.  

Actually this is a Gibbs sampler so it generates draws from the probability
distribution of possible topics given the corpus.
"""

import re
from itertools import *

import sys
import numpy
from collections import Counter, defaultdict

from heapq import nlargest
import math
import codecs

import copy

cimport numpy
cimport cython.view

class CorpusDictionary(object):
    """Convert tokens to integers and back in a streaming manner
    """
    def __init__(self, tokenizer=None, lowercase=True):
        # word -> integer
        self.token_ids = {}
        # number of times token seen
        self.counts = Counter()
        # number of docs seen in
        self.docfreq = Counter()
        # total docs seen (for computing idf)
        self.total_docs = 0
        # Convert to lowercase?
        self.lowercase = lowercase
        # integer -> word
        self.tokens = {}

        # This will match words like "won't"
        if tokenizer is None:
            self.tokenizer = re.compile(
                    r"\b\w[\w-]+[\w'-]?\w\b", re.I | re.U)
        else:
            self.tokenizer = tokenizer

    def _add_filelike(self, filelike):
        """Add the tokens in a file-like object to the dictionary"""
        self.add_string(filelike.read())

    def add_file(self, filelike_or_filename):
        """Add the tokens in a file-like object or file (with specified name) 
        to the dictionary"""
        if isinstance(filelike_or_filename, basestring):
            # codecs.open so we can handle unicode
            with codecs.open(filelike_or_filename, "r", 'utf-8') as h:
                self._add_filelike(h)
        else:
            self._add_filelike(filelike_or_filename)
    
    def add_string(self, stringlike):
        """Add tokens from stringlike to the dictionary"""
        self.add_tokens(self.tokenizer.findall(stringlike))

    def add_tokens(self, tokeniter):
        """Add a sequence of tokens, update all counts"""
        tokencounts = Counter()

        for token in tokeniter:
            if self.lowercase:
                token = token.lower()

            if token not in self.token_ids:
                token_id = len(self.token_ids)
                self.token_ids[token] = token_id
                self.tokens[token_id] = token
            
            tokencounts[token] += 1

        for token, count in tokencounts.iteritems():
            self.counts[token] += count
            self.docfreq[token] += 1

        self.total_docs += 1
    
    def trim_docfreq(self, max_df=0.8, min_df=0.01):
        """Throw out tokens according to their IDF (following 
        sklearn.feature_extraction.text"""
        ftot = float(self.total_docs)

        for token in self.token_ids.keys():
            df = self.docfreq[token]/ftot
            if df < min_df or df > max_df:
                del self.tokens[self.token_ids[token]]
                del self.token_ids[token]
                del self.docfreq[token]
                del self.counts[token]

    def tokenize_string(self, stringlike):
        """Generate a list of tokens *that are in this dictionary*"""
        return (t for t in self.tokenizer.findall(stringlike)
                    if t in self.token_ids)

    def tokenize_file(self, filelike_or_filename):
        """Generate a list of tokens *that are in this dictionary*"""
        if isinstance(filelike_or_filename, basestring):
            with codecs.open(filelike_or_filename, "r", 'utf-8') as h:
                return self.tokenize_string(h.read())
        else:
            return self.tokenize_string(filelike_or_filename.read())

    def tokens_to_ids(self, tokeniter):
        """convert a sequence of tokens (not a string) to integers"""
        return (self.token_ids[token] for token in tokeniter)

    def ids_to_tokens(self, iditer):
        """convert a sequence of integers to tokens"""
        return (self.tokens[id] for tid in iditer)

    def topwords(self, stringlike):
        """Return a list of the highest weight words ranked by tfidf"""
        ftot = self.total_docs+1.0
        token_tfidfs = ((token, count*math.log(ftot/self.docfreq[token]))
                    for token, count in Counter(
                        self.tokenize_string(stringlike)).iteritems())

        return sorted(token_tfidfs, key=lambda _: _[1], reverse=True)

cdef extern from "sparselda.h":
    ctypedef struct TopicModelParams:
        unsigned int T
        unsigned int V
        unsigned int D
        double *alpha

    void free_topic_model_params(TopicModelParams *tmp)

    int new_topic_model_params(TopicModelParams *tmp, unsigned int T)

    int add_document(
            TopicModelParams *tmp, unsigned int *tokens, unsigned int length)

    void do_iteration(TopicModelParams *tmp)

    void flat_sample_topics(TopicModelParams tmp, double *topics)

    void optimize_alpha(TopicModelParams tmp)

    void remove_documents(TopicModelParams *tmp)

    void flat_sample_documents(TopicModelParams tmp, double *doctopics)

cdef class CSparseLdaModel:
    cdef TopicModelParams tmp
    cdef public unsigned int T
    cdef public double[:] alpha
    def __cinit__(self, num_topics, *args, **kwds):
        self.T = num_topics
        new_topic_model_params(&self.tmp, num_topics)

    def __init__(self, num_topics):
        self.alpha = numpy.asarray(<numpy.double_t[:num_topics]> self.tmp.alpha)

    def add_document(self, tokiter):
        doc = list(tokiter)
        cdef unsigned int[:] adoc = numpy.asarray(doc, dtype=numpy.uint32)
        add_document(&self.tmp, &adoc[0], len(doc))

    def add_documents(self, dociter):
        for tokiter in dociter:
            self.add_document(tokiter)

    def do_iteration(self):
        do_iteration(&self.tmp)

    def update_alpha(self):
        optimize_alpha(self.tmp)

    def topics(self, dictionary):
        """Return a list of topics

        The topics will be of the form:
            [("word", weight), ("word", weight), ... ]
        sorted by weight
        """
        cdef double[:,:] topics = numpy.zeros((self.tmp.T, self.tmp.V), 
                dtype=numpy.double)
        flat_sample_topics(self.tmp, &topics[0,0])

        topiclist = []
        for t in range(0, self.tmp.T):
            topic = []
            for i in range(0, self.tmp.V):
                if topics[t,i] > 0:
                    topic.append((dictionary.tokens[i], topics[t,i]))
            topic.sort(key=lambda _:_[1], reverse=True)
            topiclist.append(topic)

        return topiclist

    def topic_matrix(self):
        """Return the topic-word matrix"""
        cdef double[:,:] topics = numpy.zeros((self.tmp.T, self.tmp.V), 
                dtype=numpy.double)
        flat_sample_topics(self.tmp, &topics[0,0])
        return topics

    def __dealloc__(self):
        free_topic_model_params(&self.tmp)

