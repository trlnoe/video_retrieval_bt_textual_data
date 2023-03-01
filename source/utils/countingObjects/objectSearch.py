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

def openJson(json_path): 
    with open(json_path) as json_file:
        data = json.load(json_file)
    return data

def indexing(writer): 
    #define field type
    t1 = FieldType()
    t1.setStored(True)
    t1.setIndexOptions(IndexOptions.DOCS)

    t2 = FieldType()
    t2.setStored(True)
    t2.setIndexOptions(IndexOptions.DOCS_AND_FREQS)
    # Open file
    data = openJson(args.json_path)

    # Add a document
    for video in data.keys(): 
        for keyframe in data[video].keys():
            doc = Document()
            doc.add(Field('video', video, t1))
            doc.add(Field('keyframe', keyframe, t1))
            text = data[video][keyframe]
            doc.add(Field('text', text, t2))
            writer.addDocument(doc)
    return writer

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
    

def main(args):
    #initialization
    env = lucene.initVM(vmargs=['-Djava.awt.headless=true'])
    #reader
    indexPath = File("AIC-Index").toPath()
    indexDir = FSDirectory.open(indexPath)
    reader = DirectoryReader.open(indexDir)
    
    #search 
    # data = openJson(args.json_path)
    search_result = luceneRetriver(args.objects, reader)
    for item in search_result: 
        print(item)
        # print('reccheck: ', data[item['video_name']][item['keyframe_id']])
    
  
#arguments
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json_path", 
        default='/workspace/competitions/AI_Challenge_2022/utils/countingObjects/objectCounting.json',
        type=str, 
        help="CLIPFeature folder",
    ), 
    parser.add_argument(
        "--objects", 
        default='Toy 3 Tree 1 Tree 1 Toy 3',
        type=str, 
        help="CLIPFeature folder",
    )
    return parser.parse_args()

    
if __name__  == "__main__":
    args = get_parser()
    main(args)


