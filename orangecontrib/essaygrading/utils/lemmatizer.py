import nltk


def lemmatizeTokens(corpus, join=False):
    lemmatizer = nltk.stem.WordNetLemmatizer()
    tokens = [[lemmatizer.lemmatize(corpus.tokens[i][j], pos=get_wordnet_pos(corpus.pos_tags[i][j]))
               for j in range(len(corpus.tokens[i]))]
              for i in range(len(corpus))]
    if join:
        tokens = [" ".join(t) for t in tokens]
    return tokens


def lemmatizeSentences(sentences, stem_ing=False):
    lemmatizer = nltk.stem.WordNetLemmatizer()
    lemmatized_sentences = []
    for s in sentences:
        if len(s) < 5:
            continue
        tokens = nltk.pos_tag(nltk.word_tokenize(s))
        # Used for ClausIE because it has some problems with -ing verbs - it crashes.
        if stem_ing:
            tokens = [token if not token[0].endswith("ing") else (token[0][:-3], token[1]) for token in tokens]
        lemmatized_sentences.append(" ".join([lemmatizer.lemmatize(token[0], pos=get_wordnet_pos(token[1]))
                                             for token in tokens]))
    return lemmatized_sentences

def stemSentences(sentences):
    stemmer = nltk.stem.PorterStemmer()
    stemmed_sentences = []
    for s in sentences:
        if len(s) < 5:
            continue
        tokens = nltk.word_tokenize(s)
        # Used for ClausIE because it has some problems with -ing verbs - it crashes.
        stemmed_sentences.append(" ".join([stemmer.stem(token) for token in tokens]))
    return stemmed_sentences


def breakToWords(s):
    charIndex = 0
    sBroken = ''
    for c in s:
        if charIndex==0:
            sBroken = sBroken + c.lower()
        elif c.isupper():
            sBroken = sBroken + ' ' + c.lower()
        else:
            sBroken = sBroken + c
        charIndex = charIndex + 1
    return(sBroken)


def get_pos_tags(s, simplify=True):
    tokens = nltk.word_tokenize(s)
    # Remove 1 letter words
    #tokens = [t for t in tokens if len(t) > 1]
    pos = nltk.pos_tag(tokens)
    if simplify:
        pos = [(p[0], get_wordnet_pos(p[1])) for p in pos]
    return pos


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
