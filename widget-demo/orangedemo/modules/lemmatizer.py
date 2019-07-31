import nltk

def lemmatizeTokens(corpus, join=False):
    lemmatizer = nltk.stem.WordNetLemmatizer()
    tokens = [[lemmatizer.lemmatize(corpus.tokens[i][j], pos=get_wordnet_pos(corpus.pos_tags[i][j]))
               for j in range(len(corpus.tokens[i]))]
              for i in range(len(corpus))]
    if join:
        tokens = [" ".join(t) for t in tokens]
    return tokens

def get_wordnet_pos(treebank_tag):

    if treebank_tag.startswith('J'):
        return nltk.corpus.wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return nltk.corpus.wordnet.VERB
    elif treebank_tag.startswith('N'):
        return nltk.corpus.wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return nltk.corpus.wordnet.ADV
    else:
        return nltk.corpus.wordnet.NOUN
