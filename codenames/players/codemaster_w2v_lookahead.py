import scipy.spatial.distance
from nltk.stem import WordNetLemmatizer
from nltk.stem.lancaster import LancasterStemmer
from math import ceil
import numpy as np
import copy
import itertools

from players.codemaster import Codemaster
THRESHOLD =  0.7 #np.inf

class AICodemaster(Codemaster):

    def __init__(self, brown_ic=None, glove_vecs=None, word_vectors=None):
        super().__init__()
        self.brown_ic = brown_ic
        self.glove_vecs = glove_vecs
        self.word_vectors = word_vectors
        self.wordnet_lemmatizer = WordNetLemmatizer()
        self.lancaster_stemmer = LancasterStemmer()
        self.cm_wordlist = []
        with open('players/cm_wordlist.txt') as infile:
            for line in infile:
                self.cm_wordlist.append(line.rstrip())
        self.root = None
        self.turn_number = 0

    def set_game_state(self, words, maps):
        if self.turn_number == 0:
            self.original_words = copy.copy(words)
            print(f"original words: {self.original_words}")
        self.words = words
        self.maps = maps
        self.update_board()
        self.init_dists()
        self.turn_number += 1

    def update_board(self):
        self.red_words = set()
        self.bad_words = set()
        self.words_guessed = set()
        for i in range(25):
            if self.words[i][0] == '*':
                self.words_guessed.add(self.original_words[i].lower())
            elif self.maps[i] == "Assassin" or self.maps[i] == "Blue" or self.maps[i] == "Civilian":
                self.bad_words.add(self.words[i].lower())
                if self.maps[i] == "Assassin":
                    self.black_word = self.words[i]
            else:
                self.red_words.add(self.words[i].lower())
    
    def init_dists(self):
        cos_dist = scipy.spatial.distance.cosine
        all_vectors = (self.word_vectors,)
        cache =  len(self.word_vectors) < 400
        self.bad_word_dists = {}
        for word in self.bad_words:
            if cache:
                    self.bad_word_dists[word] = self.word_vectors['dict_word_dists'][word]
            else:
                self.bad_word_dists[word] = {}
                for val in self.cm_wordlist:
                    b_dist = cos_dist(self.concatenate(val, all_vectors), self.concatenate(word, all_vectors))
                    self.bad_word_dists[word][val] = b_dist

        self.red_word_dists = {}
        for word in self.red_words:
            if cache:
                    self.red_word_dists[word] = self.word_vectors['dict_word_dists'][word]
            else:
                self.red_word_dists[word] = {}
                for val in self.cm_wordlist:
                    b_dist = cos_dist(self.concatenate(val, all_vectors), self.concatenate(word, all_vectors))
                    self.red_word_dists[word][val] = b_dist

    def get_clue(self):
        #self.all_guesses = set()
        if self.root is None or self.root.words_guessed != self.words_guessed:
            if self.root:
                print("board mismatch: initializing new root")
                print(f"game's words guessed: {self.words_guessed} nodes' words guessed: {self.root.words_guessed}")
            self.root = Node(self, copy.copy(self.words_guessed), None, depth = self.turn_number-1)
        self.root.get_val()
        best_clue = self.root.best_clue
        print('chosen_clue is:', best_clue[0])

        self.root = self.root.best_child
        return best_clue

    def arr_not_in_word(self, word, arr):
        if word in arr:
            return False
        lemm = self.wordnet_lemmatizer.lemmatize(word)
        lancas = self.lancaster_stemmer.stem(word)
        for i in arr:
            if i == lemm or i == lancas:
                return False
            if i.find(word) != -1:
                return False
            if word.find(i) != -1:
                return False
        return True

    def combine(self, words, wordvecs):
        factor = 1.0 / float(len(words))
        new_word = self.concatenate(words[0], wordvecs) * factor
        for word in words[1:]:
            new_word += self.concatenate(word, wordvecs) * factor
        return new_word

    def concatenate(self, word, wordvecs):
        concatenated = wordvecs[0][word]
        for vec in wordvecs[1:]:
            concatenated = np.hstack((concatenated, vec[word]))
        return concatenated


class Node:
    def __init__(self, codemaster, words_guessed, parent, depth = 0, best=np.inf):
        self.codemaster = codemaster
        self.words_guessed = words_guessed
        self.parent = parent
        self.depth = depth
        self.best_clue = None
        self.best_child = None
        self.val = np.inf
        self.terminal = False
        self.best = best

    def get_best_clues(self):
        bests = {}
        possible = {}
        cm = self.codemaster
        red_words = cm.red_words.difference(self.words_guessed)
        bad_words = cm.bad_words.difference(self.words_guessed)
        print(f"calculating best clues")
        #print(f"red word dists: {self.red_word_dists}")
        for clue_num in range(1, 3 + 1):
            best_per_dist = np.inf
            best_per = ''
            best_red_word = ''
            for red_word in list(itertools.combinations(red_words, clue_num)):
                best_word = ''
                best_dist = np.inf
                cache =  len(cm.word_vectors) < 400

                # get intersection of top X words from each red word to use as list of possible hint words
                all_top_n_words = [cm.word_vectors['sorted_word_dists'][red][:200] for red in red_word]
                if clue_num == 1:
                    possible_words = all_top_n_words[0]
                else:
                    possible_words = all_top_n_words[0]
                    for i in range(1, clue_num):
                        possible_words = [value for value in possible_words if value in all_top_n_words[i]]

                if cache:
                    for word in possible_words:
                        if not cm.arr_not_in_word(word, red_words.union(bad_words)):
                            continue

                        bad_dist = np.inf
                        worst_bad = ''
                        for bad_word in bad_words:
                            if cm.bad_word_dists[bad_word][word] < bad_dist:
                                bad_dist = cm.bad_word_dists[bad_word][word]
                                worst_bad = bad_word
                        worst_red = 0
                        for red in red_word:
                            dist = cm.red_word_dists[red][word]
                            if dist > worst_red:
                                worst_red = dist

                        if worst_red < best_dist and worst_red < bad_dist:
                            best_dist = worst_red
                            best_word = word
                            # print(worst_red,red_word,word)

                            if best_dist < best_per_dist:
                                best_per_dist = best_dist
                                best_per = best_word
                                best_red_word = red_word

                else:
                    for word in cm.cm_wordlist:
                        if not cm.arr_not_in_word(word, red_words.union(bad_words)):
                            continue

                        bad_dist = np.inf
                        worst_bad = ''
                        for bad_word in bad_words:
                            if cm.bad_word_dists[bad_word][word] < bad_dist:
                                bad_dist = cm.bad_word_dists[bad_word][word]
                                worst_bad = bad_word
                        worst_red = 0
                        for red in red_word:
                            dist = cm.red_word_dists[red][word]
                            if dist > worst_red:
                                worst_red = dist

                        if worst_red < best_dist and worst_red < bad_dist:
                            best_dist = worst_red
                            best_word = word
                            # print(worst_red,red_word,word)

                            if best_dist < best_per_dist:
                                best_per_dist = best_dist
                                best_per = best_word
                                best_red_word = red_word
                if best_dist < THRESHOLD or clue_num == 1:            
                    possible[(best_word, clue_num)] = (red_word, best_dist)
            bests[clue_num] = (best_red_word, best_per, best_per_dist)
        print(f"length of possibilities: {len(possible)}")
        return possible

    def add_children(self):
        cos_dist = scipy.spatial.distance.cosine
        cm = self.codemaster
        all_vectors = (cm.word_vectors,)
        print(f"at depth {self.depth}")
        bests = self.get_best_clues()
        for clue, clue_info in bests.items():
            combined_clue, clue_num = clue
            best_red_word, combined_score = clue_info
            worst = -np.inf
            for word in best_red_word:
                dist = cos_dist(cm.concatenate(word, all_vectors), cm.concatenate(combined_clue, all_vectors))
                if dist > worst:
                    worst = dist
            if worst < 0.7 and worst != -np.inf or clue_num == 1:
                print(f"adding clue: {clue}")
                self.add_child(clue, best_red_word)
        
    def check_board(self):
        cm = self.codemaster
        self.black_guessed = cm.black_word in self.words_guessed
        red_words = cm.red_words.difference(self.words_guessed)

        red_count = len(red_words)
        if self.black_guessed:
            self.val = np.inf
            self.terminal = True
        elif red_count == 0:
            self.val = self.depth
            self.terminal = True
            print(f"Terminal Node: depth: {self.depth}")
        else:
            self.val = 25
     
    def new_child(self, expected_words_chosen):
        new_words_guessed = copy.copy(self.words_guessed)
        for word in expected_words_chosen:
            new_words_guessed.add(word)
        return Node(self.codemaster, new_words_guessed, self, self.depth + 1, self.best)
        
    def get_val(self, depth=np.inf):
        # if self.words_guessed in self.codemaster.all_guesses:
        #     print("Board State already explored")
        #     return self.val
        # self.codemaster.all_guesses.add(self.words_guessed)
        self.check_board()
        if self.not_possible():
            print("Skipped")
            return self.val
        if self.terminal:
            if self.val < self.best:
                self.best = self.val
            return self.val
        if self.best_clue is not None:
            return self.val
        best_val = np.inf
        possible = self.get_best_clues()
        for clue, clue_info in sorted(possible.items(), key = lambda x: (x[0][1],-x[1][1]), reverse=True):
            combined_clue, clue_num = clue
            best_red_word, combined_score = clue_info
            if self.check_clue_feasible(clue_num, combined_score):
                print(f"Exploring child, depth: {self.depth+1}, clue: {clue}, dist: {combined_score}")
                child = self.new_child(best_red_word)
                child_val = child.get_val(depth)
                if child_val < best_val:
                    best_val = child_val
                    self.best_clue = clue
                    self.best_child = child
                if child.best < self.best:
                    print(f"Found new best, prev: {self.best} new: {child.best}")
                    self.best = child.best
        self.val = best_val
        return self.val

    # def best_child(self):
    #     best_clue = self.best_clue
    #     for child_key in self.children.keys():
    #         if child_key == best_clue:
    #             best_child = self.children[child_key]
    #     best_child.reset_depth()
    #     return best_child

    def not_possible(self):
        red_words = self.codemaster.red_words.difference(self.words_guessed)
        best_possible = self.depth + ceil(len(red_words)/3)
        print(f"BEST POSSIBLE: {best_possible}")
        return self.best <= best_possible or self.depth >= self.best or (not self.terminal and self.depth == self.best - 1)

    def check_clue_feasible(self, clue_num, combined_score):
        return clue_num == 1 or combined_score < THRESHOLD
        # cos_dist = scipy.spatial.distance.cosine
        # cm = self.codemaster
        # all_vectors = (cm.glove_vecs,)
        # worst = -np.inf
        # for word in best_red_word:
        #    dist = cos_dist(cm.concatenate(word, all_vectors), cm.concatenate(combined_clue, all_vectors))
        #    if dist > worst:
        #        worst = dist
        # return worst < 0.7 and worst != -np.inf or clue_num == 1


        
