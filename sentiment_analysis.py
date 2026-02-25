from pyspark.sql import SparkSession
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import string
from nltk.corpus import stopwords

def removeStopWordsFunct(x):
    stop_words=set(stopwords.words('english'))
    filteredSentence = [w for w in x if not w in stop_words]
    return filteredSentence


def removePunctuationsFunct(x):
    list_punct=list(string.punctuation)
    filtered = [''.join(c for c in s if c not in list_punct) for s in x] 
    filtered_space = [s for s in filtered if s] #remove empty space 
    return filtered_space


def lemmatizationFunct(x):
    nltk.download('wordnet')
    lemmatizer = WordNetLemmatizer()
    finalLem = [lemmatizer.lemmatize(s) for s in x]
    return finalLem


def joinTokensFunct(x):
    x = " ".join(x)
    return x


#extracting the keywords
def extractphraseFunct(x):
    
    stop_words=set(stopwords.words('english'))
    def leaves(tree):
        #Finds NP (nounphrase) leaf nodes of a chunk tree
        for subtree in tree.subtrees(filter = lambda t: t.label()=='NP'):
            yield subtree.leaves()
    def acceptable_word(word):
        """Checks conditions for acceptable word: length, stopword. We can increase the length if we want to consider large phrase"""
        accepted = bool(3 < len(word) <= 20
            and 'https' not in word.lower()
            and 'http' not in word.lower()
            and '#' not in word.lower()
            )
        yield accepted
    def get_terms(tree):
        for leaf in leaves(tree):
            term = [w for w,t in leaf if not w in stop_words if acceptable_word(w)]
            yield term
    sentence_re = r'(?:(?:[A-Z])(?:.[A-Z])+.?)|(?:\w+(?:-\w+)*)|(?:\$?\d+(?:.\d+)?%?)|(?:...|)(?:[][.,;"\'?():-_`])'
    grammar = r"""
    NBAR:
        {<NN.*|JJ>*<NN.*>}  # Nouns and Adjectives, terminated with Nouns

    NP:
        {<NBAR>}
        {<NBAR><IN><NBAR>}  # Above, connected with in/of/etc...
    """
    chunker = nltk.RegexpParser(grammar)
    tokens = nltk.regexp_tokenize(x,sentence_re)
    postoks = nltk.tag.pos_tag(tokens) #Part of speech tagging
    tree = chunker.parse(postoks) #chunking
    terms = get_terms(tree)
    temp_phrases = []
    for term in terms:
        if len(term):
            temp_phrases.append(' '.join(term))

    finalPhrase = [w for w in temp_phrases if w] #remove empty lists
    return finalPhrase


#sentiment of each key phrase
def sentimentWordsFunct(x):
    
    analyzer = SentimentIntensityAnalyzer() 
    senti_list_temp = []    
    for i in x:
        y = ''.join(i) 
        vs = analyzer.polarity_scores(y)
        senti_list_temp.append((y, vs))
        senti_list_temp = [w for w in senti_list_temp if w]    
    sentiment_list  = []
    for j in senti_list_temp:
        first = j[0]
        second = j[1]
    
        for (k,v) in second.items():
            if k == 'compound':
                if v < 0.0:
                    sentiment_list.append((first, "Negative"))
                elif v == 0.0:
                    sentiment_list.append((first, "Neutral"))
                else:
                    sentiment_list.append((first, "Positive"))     
    
    return sentiment_list


def main():

    spark = SparkSession.builder.appName("project").getOrCreate()

    output = "TermProjectResults"
    df = spark.read.json('Musical_Instruments_5.json')
    df.show(5)



    # text preprocessing
    reviews_rdd = df.select("reviewText").rdd.flatMap(lambda x: x)
    lowerCase_sentRDD = reviews_rdd.map(lambda x : x.lower())
    sentenceTokenizeRDD = lowerCase_sentRDD.map(lambda x: nltk.sent_tokenize(x))
    wordTokenizeRDD = sentenceTokenizeRDD.map(lambda x: [word for line in x for word in line.split()])



    stopwordRDD = wordTokenizeRDD.map(removeStopWordsFunct)
    rmvPunctRDD = stopwordRDD.map(removePunctuationsFunct)
    lem_wordsRDD = rmvPunctRDD.map(lemmatizationFunct)
    joinedTokens = lem_wordsRDD.map(joinTokensFunct)

    extractphraseRDD = joinedTokens.map(extractphraseFunct)





    sentimentRDD = extractphraseRDD.map(sentimentWordsFunct)
    pos_sentimentRDD = sentimentRDD.flatMap(lambda list: list).filter(lambda y:y[1]=="Positive")
    neg_sentimentRDD = sentimentRDD.flatMap(lambda list: list).filter(lambda y:y[1]=="Negative")
    neutral_sentimentRDD = sentimentRDD.flatMap(lambda list: list).filter(lambda y:y[1]=="Neutral")

    freqDistRDD = neutral_sentimentRDD.map(lambda x: (x[0], 1)).reduceByKey(lambda x,y : x+y).sortBy(lambda x: x[1], ascending = False)

    df_fDist = freqDistRDD.toDF() #converting RDD to spark dataframe
    df_fDist.createOrReplaceTempView("myTable") 
    df2 = spark.sql("SELECT _1 AS Keywords, _2 as Frequency from myTable limit 20")
    df2.coalesce(1).write.mode("overwrite").options(header=True).csv(output + "/neutral")

    freqDistRDD = pos_sentimentRDD.map(lambda x: (x[0], 1)).reduceByKey(lambda x,y : x+y).sortBy(lambda x: x[1], ascending = False)

    df_fDist = freqDistRDD.toDF() #converting RDD to spark dataframe
    df_fDist.createOrReplaceTempView("myTable") 
    df2 = spark.sql("SELECT _1 AS Keywords, _2 as Frequency from myTable limit 20") #renaming columns 
    df2.coalesce(1).write.mode("overwrite").options(header=True).csv(output + "/positive")

    freqDistRDD = neg_sentimentRDD.map(lambda x: (x[0], 1)).reduceByKey(lambda x,y : x+y).sortBy(lambda x: x[1], ascending = False)

    df_fDist = freqDistRDD.toDF() #converting RDD to spark dataframe
    df_fDist.createOrReplaceTempView("myTable") 
    df2 = spark.sql("SELECT _1 AS Keywords, _2 as Frequency from myTable limit 20") #renaming columns 
    df2.coalesce(1).write.mode("overwrite").options(header=True).csv(output + "/negative")


if __name__ == '__main__':
    main()