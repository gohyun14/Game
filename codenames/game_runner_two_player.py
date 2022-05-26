import time
import json

from game import Game
from game_two_player import GameTwoPlayer
from players.codemaster_glove_07 import AICodemaster as cm_glv07
from players.codemaster_glove_lookahead import AICodemaster as cm_glv_la
from players.codemaster_w2v_07 import AICodemaster as cm_w2v07
from players.codemaster_w2v_lookahead import AICodemaster as cm_w2v_la
from players.guesser_glove import AIGuesser as g_glv
from players.guesser_w2v import AIGuesser as g_w2v

class GameRunnerTwoPlayer:
    """Example of how to share vectors, pass kwargs, and call Game directly instead of by terminal"""

    start_time = time.time()
    glove_50d = Game.load_glove_vecs("cache_files/cache_glove_50d.txt")
    print(f"{time.time() - start_time:.2f}s to load glove50d")
    start_time = time.time()
    w2v = Game.load_w2v("cache_files/cache_w2v.txt")
    print(f"{time.time() - start_time:.2f}s to load w2v")

    print("\nclearing results folder...\n")
    Game.clear_results()
    cms = [cm_glv07, cm_glv_la, cm_w2v07]
    guessers = {cm_glv07: g_glv, cm_glv_la: g_glv, cm_w2v07: g_w2v, cm_w2v_la: g_w2v}
    with open("seeds.txt", "r") as f:
      seeds = list(map(lambda x: int(x), f.readlines()))

    start_time = time.time()
    print("starting games")
    cm_kwargs = {"glove_vecs": glove_50d, "word_vectors": w2v}
    g_kwargs = {"glove_vecs": glove_50d, "word_vectors": w2v}
    for i in range(100):
      for cm1 in cms:
        for cm2 in cms:
          GameTwoPlayer(cm1, guessers[cm1], cm2, guessers[cm2], seed=seeds[i], do_print=False,  game_name=f"two_player_exp_{seeds[i]}", cm_kwargs=cm_kwargs, g_kwargs=g_kwargs).run()
      #Game(cm_glv_la, g_glv, seed=0, do_print=True,  game_name="glv_la-glv", cm_kwargs=cm_kwargs, g_kwargs=g_kwargs).run()
    #Game(cm_glv_la, g_glv, seed=1, do_print=True,  game_name="glv_la-glv", cm_kwargs=cm_kwargs, g_kwargs=g_kwargs).run()
    #print(f"{time.time() - start_time:.2f}s to play glove lookahead")

    # display the results
    #print(f"\nfor seeds {seeds[:10]} ~")
    with open("results/bot_results_new_style_two_player.txt") as f:
        for line in f.readlines():
            game_json = json.loads(line.rstrip())
            game_name = game_json["game_name"]
            game_time = game_json["time_s"]
            game_score = game_json["total_turns"]

            print(f"time={game_time:.2f}, turns={game_score}, name={game_name}")


if __name__ == "__main__":
    GameRunnerTwoPlayer()
