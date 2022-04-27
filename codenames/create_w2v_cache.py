import numpy as np
import scipy.spatial.distance
import gensim.models.keyedvectors as word2vec
import json

def run():
  cos_dist = scipy.spatial.distance.cosine

  #load w2v 
  word_vectors = word2vec.KeyedVectors.load_word2vec_format('players/GoogleNews-vectors-negative300.bin', binary=True, unicode_errors='ignore')

  #load board words
  game_words = []
  with open("game_wordpool.txt", "r") as infile:
    for line in infile:
      game_words.append(line.rstrip().lower())

  #load cm word list
  cm_wordlist = []
  with open('players/cm_wordlist.txt') as infile:
    for line in infile:
      cm_wordlist.append(line.rstrip())

  all_vectors = (word_vectors,)

  word_dists = {}
  for word in game_words:
    word_dists[word] = {}
    for val in cm_wordlist:
      b_dist = cos_dist(concatenate(val, all_vectors), concatenate(word, all_vectors))
      word_dists[word][val] = b_dist

  with open("cache_files/cache_w2v.txt", "a") as f:
    f.write(json.dumps(word_dists))

def concatenate(word, wordvecs):
  concatenated = wordvecs[0][word]
  for vec in wordvecs[1:]:
    concatenated = np.hstack((concatenated, vec[word]))
  return concatenated

run()