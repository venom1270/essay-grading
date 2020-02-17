from flair.embeddings import WordEmbeddings, DocumentPoolEmbeddings
from flair.data import Corpus, Sentence
import numpy as np

sent = Sentence("Grass is green .")
sent2 = Sentence("Hello my green .")

text = "certain material be remove from library such a book , music and magazine , shouldn ' t be remove from the library . it give people a chance to understand how the real world @ caps2 . have certain material such a book and music definitly should not be remove , because most book and music can show most people how bad the statement in the book @ caps2 or how bad the lyric be in a song , and help that person to avoid that type of thing that the book or song @ caps2 say to the reader or listener . people should give every type"

sent = Sentence(text)

glove = DocumentPoolEmbeddings([WordEmbeddings("glove")])

glove.embed(sent)


print(sent.get_embedding())

print(sent.get_embedding().detach().numpy())




exit()

for token in sent:
    print(token)
    print(token.embedding)
