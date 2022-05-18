from tkinter import W
import numpy as np
import scipy.spatial.distance
import json

def run():
  cos_dist = scipy.spatial.distance.cosine
  #load glove vectors
  glove_vecs = {}
  with open('players/glove.6B.50d.txt', encoding="utf-8") as infile:
    for line in infile:
      line = line.rstrip().split(' ')
      glove_vecs[line[0]] = np.array([float(n) for n in line[1:]])

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

  all_vectors = (glove_vecs,)

  # make cache using dictionary of dictionaries
  dict_word_dists = {}
  for word in game_words:
    dict_word_dists[word] = {}
    for val in cm_wordlist:
      b_dist = cos_dist(concatenate(val, all_vectors), concatenate(word, all_vectors))
      dict_word_dists[word][val] = b_dist

  # make cache using dictionary of lists, for sorted word distances
  sorted_word_dists = {}
  for word in game_words:
    sorted_word_dists[word] = []
    for val in cm_wordlist:
      b_dist = cos_dist(concatenate(val, all_vectors), concatenate(word, all_vectors))
      sorted_word_dists[word].append((val, b_dist))
    sorted_word_dists[word] = sorted(sorted_word_dists[word], key = lambda x: x[1])
    sorted_word_dists[word] = [tup[0] for tup in sorted_word_dists[word][1:]]

  word_dists = {}
  word_dists['dict_word_dists'] = dict_word_dists
  word_dists['sorted_word_dists'] = sorted_word_dists
  
  with open("cache_files/cache_glove_50d.txt", "a") as f:
    f.write(json.dumps(word_dists))

def concatenate(word, wordvecs):
  concatenated = wordvecs[0][word]
  for vec in wordvecs[1:]:
    concatenated = np.hstack((concatenated, vec[word]))
  return concatenated

run()