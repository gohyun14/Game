from nltk.corpus import wordnet
from nltk.corpus import words
from nltk.corpus import wordnet_ic
from operator import itemgetter
from players.guesser import guesser
from collections import Counter
from allennlp.commands.elmo import ElmoEmbedder
import gensim.models.keyedvectors as word2vec
import gensim.downloader as api
import itertools
import numpy as np
import random
import scipy


class wn_guesser(guesser):


    def __init__(self):
        # self.word_vectors = api.load("glove-wiki-gigaword-100")
        self.brown_ic = wordnet_ic.ic('ic-brown.dat')
        self.word_vectors = word2vec.KeyedVectors.load_word2vec_format(
            'players/GoogleNews-vectors-negative300.bin', binary=True, unicode_errors='ignore')
        self.glove_vecs = {}
        with open('players/glove/glove.6B.300d.txt') as infile:
            for line in infile:
                line = line.rstrip().split(' ')
                self.glove_vecs[line[0]] = np.array([float(n) for n in line[1:]])
        # self.elmo = ElmoEmbedder()
        

    def get_board(self, words):
        self.words = words
        return words


    def get_clue(self, clue, num):
        self.clue = clue
        print("The clue is:", clue, num, sep=" ")

        li = [clue, num]
        return li


    def wordnet_synset(self, clue, board):
        pat_results = []
        jcn_results = []
        lin_results = []
        count = 0

        for i in (board):
            for clue_list in wordnet.synsets(clue):
                pat_clue = jcn_clue = lin_clue = 0
                for board_list in wordnet.synsets(i):
                    try:
                        # only if the two compared words have the same part of speech
                        pat = clue_list.path_similarity(board_list)
                        jcn = clue_list.jcn_similarity(board_list, self.brown_ic)
                        lin = clue_list.lin_similarity(board_list, self.brown_ic)
                    except:
                        continue

                    if jcn:
                        jcn_results.append(("jcn: ", jcn, count, clue_list, board_list, i))
                        lin_results.append(("lin: ", lin, count, clue_list, board_list, i))
                        if jcn > jcn_clue:
                            jcn_clue = jcn

                    if pat:
                        pat_results.append(("pat: ", pat, count, clue_list, board_list, i))
                        if pat > pat_clue:
                            pat_clue = pat 

        # if results list is empty
        if not jcn_results:
            return []
        
        pat_results = list(reversed(sorted(pat_results, key=itemgetter(1))))
        lin_results = list(reversed(sorted(lin_results, key=itemgetter(1))))
        jcn_results = list(reversed(sorted(jcn_results, key=itemgetter(1))))

        results = [pat_results[:3], jcn_results[:3], lin_results[:3]]
        return results


    def compute_GooGlove(self, clue, board):
        w2v = []
        glove = []
        linalg_result = []

          # load pre-trained word-vectors from gensim-data
        # result = word_vectors.most_similar(positive=['woman', 'king'], negative=['man'])
        # print("{}: {:.4f}".format(*result[0])) # queen: 0.7699
        #
        # result = word_vectors.most_similar_cosmul(positive=['woman', 'king'], negative=['man'])
        # print("{}: {:.4f}".format(*result[0])) # queen: 0.8965
        #
        # print(word_vectors.doesnt_match("breakfast cereal dinner lunch".split())) # cereal
        #
        # similarity = word_vectors.similarity('woman', 'man')
        # similarity > 0.8 # True
        #
        # result = word_vectors.similar_by_word("cat")
        # print("{}: {:.4f}".format(*result[0])) # dog: 0.8798
        #
        # sentence_obama = 'Obama speaks to the media in Illinois'.lower().split()
        # sentence_president = 'The president greets the press in Chicago'.lower().split()
        #
        # similarity = word_vectors.wmdistance(sentence_obama, sentence_president)
        # print("{:.4f}".format(similarity)) # 3.4893
        #
        # distance = word_vectors.distance("media", "media")
        # print("{:.1f}".format(distance)) # 0.0
        #
        # sim = word_vectors.n_similarity(['sushi', 'shop'], ['japanese', 'restaurant'])
        # print("{:.4f}".format(sim)) # 0.7067
        #
        # vector = word_vectors['computer']  # numpy vector of a word
        # vector.shape # (100,)
        #
        # vector = word_vectors.wv.word_vec('office', use_norm=True)
        # vector.shape # (100,)

        for word in board:
            try:
                if word[0] == '*':
                    continue

                # for i in range(25):
                #     linalg = np.dot(words[board[i].lower()] / np.linalg.norm(words[board[i].lower()]),
                #         words[clue.lower()] / np.linalg.norm(words[clue.lower()]))
                #     linalg_result.append([board[i], clue, linalg])

                w2v.append((scipy.spatial.distance.cosine(self.word_vectors[clue], 
                    self.word_vectors[word.lower()]),word))
                glove.append((scipy.spatial.distance.cosine(self.glove_vecs[clue],
                    self.glove_vecs[word.lower()]),word))

            except KeyError:
                continue

        print("w2v ", sorted(w2v)[:1])
        print("glove ", sorted(glove)[:1])

        w2v = list(sorted(w2v))
        glove = list(sorted(glove))
        # linalg_result = list(reversed(sorted(linalg_result, key=self.take_third)))

        result = w2v[:3] + glove[:3]
        return result


    def give_answer(self):
        weights = [15, 12, 8, 8, 8]

        sorted_results = self.wordnet_synset(self.clue, self.words)
        google_glove = self.compute_GooGlove(self.clue, self.words)

        if google_glove and sorted_results:
            #w2v
            if(google_glove[0][0] < 0.8):
                if(google_glove[0][0] < 0.7):
                    if(google_glove[0][0] < 0.51):
                        weights[0] += 20
                    weights[0] += 10
                weights[0] += 3
            #glove
            if(google_glove[3][0] < 0.66):
                if(google_glove[3][0] < 0.51):
                    if(google_glove[3][0] < 0.36):
                        weights[1] += 20
                    weights[1] += 10
                weights[1] += 4
            #path_sim
            if(sorted_results[0][0][1] > 0.24):
                if(sorted_results[0][0][1] > 0.34):
                    if(sorted_results[0][0][1] > 0.49):
                        weights[2] += 20
                    weights[2] += 10
                weights[2] += 5
            #jcn_sim
            if(sorted_results[1][0][1] > 0.10):
                if(sorted_results[1][0][1] > 0.128):
                    if(sorted_results[1][0][1] > 0.19):
                        weights[3] += 20
                    weights[3] += 10
                weights[3] += 5
            #lin_sim
            if(sorted_results[2][0][1] > 0.52):
                if(sorted_results[2][0][1] > 0.64):
                    if(sorted_results[2][0][1] > 0.79):
                        weights[4] += 20
                    weights[4] += 10
                weights[4] += 5

            for i in [i[0] for i in sorted_results]:
                print(i)
            # google_glove[0][1] is w2v scipy cosine value

            maxWeight = max(weights)
            y = ([i for i, j in enumerate(weights) if j == maxWeight])
            x = int(y[0])
            print(x,y)
            if x == 0:
                string_answer_input = (google_glove[0][1])
            elif x == 1:
                string_answer_input = (google_glove[3][1])
            elif x == 2:
                string_answer_input = (sorted_results[0][0][5])
            elif x == 3:
                string_answer_input = (sorted_results[1][0][5])
            elif x == 4:
                string_answer_input = (sorted_results[2][0][5])

        else:
            return("no comparisons")
        
        print("Threshold chose word: ", string_answer_input)
        return string_answer_input


