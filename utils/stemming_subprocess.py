from nltk.stem import PorterStemmer
stemmer = PorterStemmer()

def prepare_query(query):
    query_words = query.strip().lower().split()
    stemmed_query_words = [stemmer.stem(word) for word in query_words] 
    stemmed_query_words = ' '.join(stemmed_query_words)
    print(stemmed_query_words)
    return stemmed_query_words