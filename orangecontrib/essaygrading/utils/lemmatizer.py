import nltk


def lemmatizeTokens(corpus, join=False):
    '''
    Lemmatize all tokens in Corpus objects and optionally join them.
    :param corpus: Corpus to lemmatize.
    :param join: join split tokens to sentence.
    :return: lemmatized tokens.
    '''
    lemmatizer = nltk.stem.WordNetLemmatizer()
    tokens = [[lemmatizer.lemmatize(corpus.tokens[i][j], pos=get_wordnet_pos(corpus.pos_tags[i][j]))
               for j in range(len(corpus.tokens[i]))]
              for i in range(len(corpus))]
    if join:
        tokens = [" ".join(t) for t in tokens]
    return tokens


def lemmatizeSentences(sentences, stem_ing=False):
    '''
    Lemmatize list of senetences.
    :param sentences: list of sentence strings.
    :param stem_ing: flag to lemmatize words ending with "ing".
    :return: lemmatized sentences.
    '''
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
    '''
    Stem list of sentences.
    :param sentences: list of sentence strings.
    :return: stemmed sentences.
    '''
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
    '''
    Break string to words.
    :param s: string.
    :return: list of words
    '''
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
    '''
    Get POS tags of sentence string.
    :param s: sentence string.
    :param simplify: flag to return similified wordnet POS tags.
    :return: list of (string, POS_tag) elements.
    '''
    tokens = nltk.word_tokenize(s)
    # Remove 1 letter words
    #tokens = [t for t in tokens if len(t) > 1]
    pos = nltk.pos_tag(tokens)
    if simplify:
        pos = [(p[0], get_wordnet_pos(p[1])) for p in pos]
    return pos


def get_wordnet_pos(treebank_tag):
    '''
    Convert form Treebank POS tag to Wordnet POS tag.
    :param treebank_tag: Treebank POS tag string.
    :return: Wordnet tag string, defaults to noun.
    '''
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
