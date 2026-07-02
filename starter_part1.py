import gensim.downloader
import numpy as np

# Pre-trained embedding model (do not modify)
model = gensim.downloader.load("word2vec-google-news-300")
EMBEDDING_DIM = model.vector_size


def replace_with_similar(sentence, indices):
    """
    Replace the tokens at the given indices with their top-1 most similar words.
    """

    words = sentence.split()
    similar_words_dict = {}
    
    for i in range(len(indices)):


        idx = indices[i]
        target = words[idx]
        # find similar words
        similars = model.most_similar(target, topn=5)
        similar_words_dict[target] = similars
        # get the best one

        best = similars[0][0]
        words[idx] = best
    new_sent = " ".join(words)

    return new_sent, similar_words_dict


def sentence_vector(sentence):
    """
    Computes sentence embedding as mean of word vectors.
    """
    vector_dict = {}
    words = sentence.split()
    total = np.zeros(EMBEDDING_DIM)

    count = 0
    for i in range(len(words)):
        w = words[i]
        if w in model:
            vec = model[w]
        else:

            vec = np.zeros(300)
        vector_dict[w] = vec
        total = total + vec
        count += 1
    
    # take the average
    result = total / count
    return vector_dict, result


def most_similar_sentences(file_path, query):
    """
    Rank sentences by cosine similarity to query.
    """
    results = []
    _, query_vec = sentence_vector(query)
    # REad Files
    f = open(file_path, "r", encoding="utf-8")
    lines = f.readlines()
    f.close()
    for i in range(len(lines)):
        line = lines[i].strip()
        if line == "":
            continue
        _, svec = sentence_vector(line)
        
        # calcuate cosine similarity
        dot = np.dot(query_vec, svec)
        norm1 = np.linalg.norm(query_vec) # ???
        norm2 = np.linalg.norm(svec)
        cos_sim = dot / (norm1 * norm2)
        
        results.append((line, cos_sim))
    
    # sort from highest to lowest
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def analyze_dimension_contributions(word1, word2, top_k=10):
    """
    Analyze dimension-wise contributions.
    """
    vec1 = model[word1]
    vec2 = model[word2]
    contribs = []
    for d in range(EMBEDDING_DIM):

        val = vec1[d] * vec2[d]
        contribs.append((d, float(val), float(vec1[d]), float(vec2[d])))
    
    # sort by absolute value
    contribs.sort(key=lambda x: abs(x[1]), reverse=True)
    topk = contribs[:top_k]

    return topk
