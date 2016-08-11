import nltk
from nltk.draw.tree import TreeView
import os

#sentence = "When did amazon start delivering coffee to Carl?"
#sentence = "I drank that milk from an old bottle"
sentence = "shit there shit covered in dust."
tokens = nltk.word_tokenize(sentence)
tagged = nltk.pos_tag(tokens)
entities = nltk.chunk.ne_chunk(tagged)
print(entities)
entities.draw()

