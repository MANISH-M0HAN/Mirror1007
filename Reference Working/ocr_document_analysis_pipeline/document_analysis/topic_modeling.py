from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

def perform_topic_modeling(text, num_topics=5):
    """Perform topic modeling on the given text."""
    try:
        vectorizer = CountVectorizer(stop_words='english')
        doc_term_matrix = vectorizer.fit_transform([text])
        lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
        lda.fit(doc_term_matrix)
        topics = {}
        for idx, topic in enumerate(lda.components_):
            topics[f"Topic {idx+1}"] = [vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-10:]]
        return topics
    except Exception as e:
        print("Error occurred during topic modeling:", e)
        return {}

# Old code below 
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.decomposition import LatentDirichletAllocation

# def perform_topic_modeling(text, num_topics=5):
#     """Perform topic modeling using Latent Dirichlet Allocation (LDA)."""
#     vectorizer = CountVectorizer(stop_words='english')
#     doc_term_matrix = vectorizer.fit_transform([text])
#     lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
#     lda.fit(doc_term_matrix)
#     topics = {}
#     for idx, topic in enumerate(lda.components_):
#         topics[f"Topic {idx+1}"] = [vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-10:]]
#     return topics
