import zipfile
import numpy as np
import gc
import os
#zip_ref = zipfile.ZipFile('/content/drive/My Drive/embeddings/embeddings.zip', 'r')

def load_glove(word_index, max_features, unk_uni, create = False):
    path = '/content/drive/My Drive/COVID19 Fake News Detection in English/neural-network/resources'
    if create == False:
        embedding_matrix = np.load(os.path.join(path,'globe_embedding.npy'))
        return embedding_matrix, None
    def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')
    EMBEDDING_FILE = zip_ref.open('glove.840B.300d/glove.840B.300d.txt', 'r')
    embeddings_index = dict(get_coefs(*o.decode().split(" "))  for o in EMBEDDING_FILE)

    all_embs = np.stack(list(embeddings_index.values()))
    emb_mean, emb_std = all_embs.mean(), all_embs.std()
    embed_size = all_embs.shape[1]

    unknown_words = []
    nb_words = min(max_features, len(word_index))

    if unk_uni:
        embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    else:
        embedding_matrix = np.zeros((nb_words+1, embed_size))
    for word, i in word_index.items():
        if i >= nb_words:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is None:
            embedding_vector = embeddings_index.get(word.lower())
            if embedding_vector is None:
                unknown_words.append((word, i))
            else:
                embedding_matrix[i] = embedding_vector
        else:
            embedding_matrix[i] = embedding_vector
    print('\nTotal unknowns glove', len(unknown_words))
    print(unknown_words[:10])
    np.save(os.path.join(path,'globe_embedding.npy'),embedding_matrix)
    del embeddings_index
    gc.collect()
    return embedding_matrix, unknown_words



def load_fasttext(word_index, max_features, unk_uni =True, create = False):
    path = '/content/drive/My Drive/COVID19 Fake News Detection in English/neural-network/resources'
    if create == False:
        embedding_matrix = np.load(os.path.join(path,'fasttext_embedding.npy'))
        return embedding_matrix, None
    def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')
    
    EMBEDDING_FILE = zip_ref.open('wiki-news-300d-1M/wiki-news-300d-1M.vec', 'r')
    embeddings_index = dict(get_coefs(*o.decode().split(" "))  for o in EMBEDDING_FILE if len(o) > 100)

    all_embs = np.stack(list(embeddings_index.values()))
    emb_mean, emb_std = all_embs.mean(), all_embs.std()
    embed_size = all_embs.shape[1]

    unknown_words = []
    nb_words = min(max_features, len(word_index))

    if unk_uni:
        embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words+1, embed_size))
    else:
        embedding_matrix = np.zeros((nb_words + 1, embed_size))
    for word, i in word_index.items():
        if i >= nb_words:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is None:
            embedding_vector = embeddings_index.get(word.lower())
            if embedding_vector is None:
                unknown_words.append((word, i))
            else:
                embedding_matrix[i] = embedding_vector
        else:
            embedding_matrix[i] = embedding_vector
    print('\nTotal unknowns wiki', len(unknown_words))
    print(unknown_words[:10])
    np.save(os.path.join(path,'fasttext_embedding.npy'),embedding_matrix)
    del embeddings_index
    gc.collect()
    return embedding_matrix, unknown_words