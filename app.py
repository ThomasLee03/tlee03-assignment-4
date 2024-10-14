from flask import Flask, render_template, request, jsonify
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import nltk
from nltk.corpus import stopwords
from numpy.linalg import svd

nltk.download('stopwords')

app = Flask(__name__)


# TODO: Fetch dataset, initialize vectorizer and LSA here

newsgroups = fetch_20newsgroups(subset='all')
documents = newsgroups.data

# Initialize the TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'), max_features=5000)
X_tfidf = vectorizer.fit_transform(documents).toarray()  # Convert sparse matrix to dense

# Step 2: Perform SVD decomposition manually
U, S, VT = svd(X_tfidf, full_matrices=False)

# Step 3: Truncate the matrices to keep only top 100 components
n_components = 100
U_reduced = U[:, :n_components]  # Left singular vectors
S_reduced = np.diag(S[:n_components])  # Truncated singular values
VT_reduced = VT[:n_components, :]  # Right singular vectors

# Step 4: Calculate the LSA representation of the documents
X_lsa = np.dot(U_reduced, S_reduced)  # Use reduced matrices for LSA

def cosine_similarity(vec1, vec2):
    """
    Compute the cosine similarity between two vectors from scratch.
    """
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0  # Handle edge cases where one of the vectors is all zeros
    return dot_product / (norm_vec1 * norm_vec2)   


def search_engine(query):
    """
    Function to search for top 5 similar documents given a query
    Input: query (str)
    Output: documents (list), similarities (list), indices (list)
    """
    # TODO: Implement search engine here
    # return documents, similarities, indices 
        # Transform the query to TF-IDF space and project it into LSA space
    # Transform the query into the TF-IDF space
    # Step 1: Transform the query into the TF-IDF space
    query_tfidf = vectorizer.transform([query]).toarray()

    # Step 2: Project the query into the LSA space
    query_lsa = np.dot(query_tfidf, VT_reduced.T)

    # Step 3: Compute cosine similarity between the query and all documents
    similarities = [cosine_similarity(query_lsa[0], doc) for doc in X_lsa]

    # Step 4: Get the indices of the top 5 most similar documents
    top_indices = np.argsort(similarities)[::-1][:5]

    # Convert top_indices to a list of plain integers
    top_indices = [int(i) for i in top_indices]

    # Step 5: Retrieve the top 5 documents and their similarities
    top_documents = [documents[i] for i in top_indices]
    top_similarities = [similarities[i] for i in top_indices]

    return top_documents, top_similarities, top_indices

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']
    documents, similarities, indices = search_engine(query)

    # Ensure all outputs are converted to standard Python types
    similarities = [float(sim) for sim in similarities]  # Convert similarities to float
    indices = [int(idx) for idx in indices]  # Convert indices to int

    # Ensure documents are strings and not any other type
    documents = [str(doc) for doc in documents]

    return jsonify({
        'documents': documents,
        'similarities': similarities,
        'indices': indices
    })


if __name__ == '__main__':
    app.run(debug=True)
