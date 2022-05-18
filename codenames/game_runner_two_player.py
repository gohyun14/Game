import time
import json

from game_two_player import GameTwoPlayer
from players.codemaster_glove_07 import AICodemaster as cm_glv07
from players.codemaster_glove_lookahead import AICodemaster as cm_glv_la
from players.guesser_glove import AIGuesser as g_glv
from players.vector_codemaster import VectorCodemaster
from players.vector_guesser import VectorGuesser
from players.codemaster_w2vglove_07 import AICodemaster as cm_w2vglv07
from players.guesser_w2vglove import AIGuesser as g_w2vglv

class GameRunnerTwoPlayer:
    """Example of how to share vectors, pass kwargs, and call Game directly instead of by terminal"""

    start_time = time.time()
    glove_50d = GameTwoPlayer.load_glove_vecs("players/glove.6B.50d.txt")
    print(f"{time.time() - start_time:.2f}s to load glove50d")

    #print("\nclearing results folder...\n")
    #GameTwoPlayer.clear_results()

    with open("seeds.txt", "r") as f:
      seeds = list(map(lambda x: int(x), f.readlines()))

    start_time = time.time()
    print("starting lookahead glove_glove game")
    cm_kwargs = {"glove_vecs": glove_50d}
    g_kwargs = {"glove_vecs": glove_50d}
    for i in range(10):
      GameTwoPlayer(cm_glv_la, g_glv, cm_glv07, g_glv, seed=seeds[i], do_print=True,  game_name="glv_la-glv", cm_kwargs=cm_kwargs, g_kwargs=g_kwargs).run()
    #Game(cm_glv_la, g_glv, seed=0, do_print=True,  game_name="glv_la-glv", cm_kwargs=cm_kwargs, g_kwargs=g_kwargs).run()
    #Game(cm_glv_la, g_glv, seed=1, do_print=True,  game_name="glv_la-glv", cm_kwargs=cm_kwargs, g_kwargs=g_kwargs).run()
    #print(f"{time.time() - start_time:.2f}s to play glove lookahead")

    # display the results
    print(f"\nfor seeds {seeds[:10]} ~")
    with open("results/bot_results_new_style_two_player.txt") as f:
        for line in f.readlines():
            game_json = json.loads(line.rstrip())
            game_name = game_json["game_name"]
            game_time = game_json["time_s"]
            game_score = game_json["total_turns"]

            print(f"time={game_time:.2f}, turns={game_score}, name={game_name}")


if __name__ == "__main__":
    GameRunnerTwoPlayer()
