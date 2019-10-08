import spacy
import neuralcoref

nlp = spacy.load("en_core_web_lg")
coref = neuralcoref.NeuralCoref(nlp.vocab)

nlp.add_pipe(coref, name="neuralcoref")

#doc = nlp('My sister has a dog. She loves him.')
#doc = nlp("Trump meanwhile escalated his attacks on the whistleblower -- demanding to meet his accuser face to face during a day of rage-filled tweets about the Democratic attempt to impeach him. The President's anger spilled over during a day of rage-filled tweets Sunday in which he selectively quoted a supporter who said he was afraid of a Civil War-like fracture in the country if Trump is forced from office.")

#doc = nlp("Lisa and Tom like sports a lot. They compete in tournaments all around the world. Lisa is better than Tom but she lacks discipline while he lacks overall skill.")

doc = nlp("My sister and brother hate each other. They fight all the time.")

print(doc._.has_coref)
print(doc._.coref_clusters)
print(doc._.coref_clusters[0].mentions)
print(doc._.coref_clusters[0].mentions[0]._.coref_cluster.main)
print(doc._.coref_resolved)
