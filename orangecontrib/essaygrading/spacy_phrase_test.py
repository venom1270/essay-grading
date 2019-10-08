import spacy


def get_pps(doc):
    "Function to get PPs from a parsed document."
    pps = []
    for token in doc:
        print(token.pos_)
        # Try this with other parts of speech for different subtrees.
        if token.pos_ == 'VERB':
            pp = ' '.join([tok.orth_ for tok in token.subtree])
            pps.append(pp)
    return pps

def get_vp(doc):
    "Function to get PPs from a parsed document."
    pps = []
    vps = []
    vp = ""
    for token in doc:
        # Try this with other parts of speech for different subtrees.
        if token.pos_ == 'VERB':
            if vp == "":
                vp = str(token)
            else:
                vp += " " + str(token)
        elif vp != "":
            vps.append(vp)
            vp = ""
    return vps


nlp = spacy.load('en')
#ex = 'A short man in blue jeans is working in the kitchen.'
ex = "I have been working my whole life and have nothing to show for it. I'm going to file a complaint to stop slavery."
sentence = "Lisa likes tennis. Lisa doesn't like tennis."
doc = nlp(sentence)

print(get_vp(doc))


import spacy,en_core_web_sm
import textacy
nlp = en_core_web_sm.load()

pattern = r'<VERB>?<ADV>*<VERB>+'
doc = nlp(sentence)
lists = textacy.extract.pos_regex_matches(doc, pattern)
for list in lists:
    print(list.text)