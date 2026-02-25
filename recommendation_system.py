from pyspark.sql import SparkSession
import re
import numpy as np
import sys

# AMAZONREVIEWSFILE = sys.argv[1]
AMAZONREVIEWSFILE = 'Musical_Instruments_5.json.gz'
NUMBEROFWORDS = 1000
productID = 'B0002D01PY'

def buildArray(listOfIndices):
  returnVal = np.zeros(NUMBEROFWORDS)
  for index in listOfIndices:
      returnVal[index] = returnVal[index] + 1
  mysum = np.sum(returnVal)
  returnVal = np.divide(returnVal, mysum)
  return returnVal

def build_one_hot_array (listOfIndices):
  returnVal = np.zeros (NUMBEROFWORDS)
  for index in listOfIndices:
      if returnVal[index] == 0: returnVal[index] = 1
  return returnVal

def cosSim (x,y):
	normA = np.linalg.norm(x)
	normB = np.linalg.norm(y)
	return np.dot(x,y)/(normA*normB)

def getPrediction(productID, k, tfidfVectors):

  # Get the tf * idf array for the input productID
  inputTFIDF = tfidfVectors.filter(lambda x: x[0] == productID).collect()

  # Get the distance from the input text string to all database documents,
  # using cosine similarity
  distances = tfidfVectors.map(lambda x: (x[0], cosSim(x[1], inputTFIDF[0][1])))

  # Get top k recommended items
  topK = distances.top(k, lambda x: x[1])[1:]

  # Strip similarity score, just return productIDs
  topK = list(zip(*topK))[0]

  return topK, inputTFIDF[0][0]

def main():
  # Create Spark Session
  spark = SparkSession.builder.appName('sparkdf').getOrCreate()
  sc = spark.sparkContext

  # Read data
  allReviewData = spark.read.json(AMAZONREVIEWSFILE).rdd
  
  # Get number of reviews
  numDocs = allReviewData.count()

  # Get just IDs and review text
  # (productID, text)
  idAndText = allReviewData.map(lambda x: (x[0], x[3]))

  # Now, we split the text in each (docID, text) pair into a list of words
  # After this step, we will have a data set with
  # (docID, ["word1", "word2", "word3", ...])

  # Remove all non alpha characters, make them lowercase, and split on spaces
  regex = re.compile('[^a-zA-Z]')
  idAndTermsList = idAndText\
                      .map(lambda x : (x[0], regex.sub(' ', x[1])\
                      .lower().split()))

  idAndTermsList = idAndTermsList.reduceByKey(lambda x,y: x+y)

  # Now get the top NUMBEROFWORDS terms
  # 
  # first change (docID, ["word1", "word2", "word3", ...])
  # to ("word1", 1) ("word2", 1)
  termOnePairs = idAndTermsList.flatMap(lambda x: x[1]).map(lambda x: (x, 1))

  # Now, count all of the terms, giving us ("word1", 1433), ("word2", 3423423), etc.
  allCounts = termOnePairs.reduceByKey(lambda x,y: x+y)

  # Get the top NUMBEROFWORDS terms in a local array in a sorted format based on frequency 
  topTerms = allCounts.takeOrdered(NUMBEROFWORDS, lambda x: -x[1])

  # print("Top 100 Terms in Corpus:")
  # for x in topTerms[:100]:
  #   print(x)

  # We'll create a RDD that has a set of (word, dictNum) pairs
  # start by creating an RDD that has the number 0 through 1000
  # 1000 is the number of terms that will be in our dictionary
  kZerosArray = sc.parallelize(range(NUMBEROFWORDS))

  # Now, we transform (0), (1), (2), ... to ("MostCommonWord", 1)
  # ("NextMostCommon", 2), ...
  # the number will be the spot in the dictionary used to tell us
  # where the word is located
  dictionary = kZerosArray.map(lambda x : (topTerms[x][0], x))



  # Next create TF vectors

  # Next, we get a RDD that has, for each (docID, ["word1", "word2", "word3", ...]),
  # ("word1", docID), ("word2", docId), ...
  termID_pairs = idAndTermsList.flatMap(lambda x: ((j, x[0]) for j in x[1]))

  # # Now join and link them, to get a set of ("word1", (dictionaryPos, docID)) pairs
  termOnePairs = dictionary.join(termID_pairs)

  # Now, we drop the actual word itself to get a set of (docID, dictionaryPos) pairs
  docPosPairs = termOnePairs.map(lambda x: (x[1][1], [x[1][0]]))

  # Now get a set of (docID, [dictionaryPos1, dictionaryPos2, dictionaryPos3...]) pairs
  termsInEachDoc = docPosPairs.reduceByKey(lambda x,y: x+y)\
                                    .map(lambda x: (x[0], sorted(x[1])))                                    

  # Convert list to numpy array
  termFreq = termsInEachDoc.map(lambda x: (x[0], buildArray(x[1])))

  # Create a OneHot encoding 
  oneHot = termsInEachDoc.map(lambda x: (x[0], build_one_hot_array(x[1])))

  # Now, add up all of those arrays into a single array, where the
  # i^th entry tells us how many
  # individual documents the i^th word in the dictionary appeared in
  docFreq = oneHot.reduce(lambda x, y: ("", np.add(x[1], y[1])))[1]

  # # Create an array of NUMBEROFWORDS entries, each entry with the value numberOfDocs (number of docs)
  # multiplier = np.full(NUMBEROFWORDS, numDocs)

  # Get the version of dfArray where the i^th entry is the inverse-document frequency for the
  # i^th word in the corpus
  inverseDocFreq = np.log(np.divide(np.full(NUMBEROFWORDS, numDocs), docFreq))

  # Finally, convert all of the tf vectors in termFreq to tf * idf vectors
  tfidfVectors = termFreq.map(lambda x: (x[0],np.multiply(x[1],inverseDocFreq)))

  # Test
  pred, item = getPrediction(productID, 20, tfidfVectors)
  print('Product recommendations for:', item)
  for x in pred:
    print(x)

if __name__ == '__main__':
  main()