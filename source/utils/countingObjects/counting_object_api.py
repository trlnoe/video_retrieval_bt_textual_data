# import lucene
# from java.io import StringReader
# from org.apache.lucene.analysis.ja import JapaneseAnalyzer
# from org.apache.lucene.analysis.standard import StandardAnalyzer, StandardTokenizer
# from org.apache.lucene.analysis.tokenattributes import CharTermAttribute
# lucene.initVM(vmargs=['-Djava.awt.headless=true'])

# # StandardAnalyzer example.
# test = "This is how we do it."
# analyzer = StandardAnalyzer()
# stream = analyzer.tokenStream("", StringReader(test))
# stream.reset()
# tokens = []
# while stream.incrementToken():
#     tokens.append(stream.getAttribute(CharTermAttribute.class_).toString())
# print(tokens)

import os
from pathlib import Path
import argparse
import lucene

from java.io import File
from java.nio.file import Paths
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.document import Document, Field, FieldType
from org.apache.lucene.index import (IndexOptions, IndexWriter,
                                     IndexWriterConfig, DirectoryReader)
from org.apache.lucene.search import IndexSearcher
from org.apache.lucene.store import MMapDirectory, FSDirectory
from org.apache.lucene.queryparser.classic import QueryParser
import json

import flask
from flask import jsonify, request
from datetime import datetime
from tqdm import tqdm
from typing import List, Tuple
import numpy as np

def openJson(json_path): 
    with open(json_path) as json_file:
        data = json.load(json_file)
    return data


def luceneRetriver(objects, reader):
    analyzer = StandardAnalyzer()
    searcher = IndexSearcher(reader)
    query = QueryParser("text", analyzer).parse(objects)
    MAX = 1000
    hits = searcher.search(query, MAX)

    print(f'Found {hits.totalHits} document(s) that matched query {query}')
    search_result = []
    
    for hit in hits.scoreDocs:
        doc = searcher.doc(hit.doc)
        
        search_result.append({"video_name":doc.get('video'),
                                "keyframe_id": doc.get('keyframe'),
                                "score": hit.score,
                                "text": doc.get('origin')})
    return search_result


app = flask.Flask("Counting Object")
app.config["DEBUG"] = True
lucene.initVM(vmargs=['-Djava.awt.headless=true'])



#API
@app.route('/predict', methods=['POST', 'GET'])
def countingSearch():
    #initialization
    #reader
    
    lucene.getVMEnv().attachCurrentThread()
    indexPath = File("AIC-Index").toPath()
    indexDir = FSDirectory.open(indexPath)
    reader = DirectoryReader.open(indexDir)

    if request.method == "POST":
        object_str = request.json['object_str']
        query = request.json['query']
    else:
        object_str = request.args.get('object_str')
        query = request.args.get('query')

    search_result = luceneRetriver(object_str, reader)

    response = flask.jsonify(search_result)
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    response.success = True
    return response  

    
if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 5000, debug=False)


