import subprocess

# solitare games
def run():
    # w2v 0.7 greedy
    counter = 100
    for i in range(30):
        str_counter = str(int(counter))
        subprocess.run(["python", "run_game.py", "players.codemaster_w2v_07.AICodemaster", "players.guesser_w2v.AIGuesser",
                        "--seed", str_counter, "--w2v", "cache_files/cache_w2v.txt", "--no_print", "--game_name", "W2V-0.7-Greedy"])
        counter += 50
    
    # glove 0.7 greedy
    counter = 100
    for i in range(30):
        str_counter = str(int(counter))
        subprocess.run(["python", "run_game.py", "players.codemaster_glove_07.AICodemaster", "players.guesser_glove.AIGuesser",
                        "--seed", str_counter, "--glove_cm", "cache_files/cache_glove_50d.txt", "--glove_guesser",
                        "cache_files/cache_glove_50d.txt", "--no_print", "--game_name", "Glove-0.7-Greedy"])
        counter += 50

    # glove lookahead
    counter = 100
    for i in range(30):
        str_counter = str(int(counter))
        subprocess.run(["python", "run_game.py", "players.codemaster_glove_lookahead.AICodemaster", "players.guesser_glove.AIGuesser",
                        "--seed", str_counter, "--glove_cm", "cache_files/cache_glove_50d.txt", "--glove_guesser",
                        "cache_files/cache_glove_50d.txt", "--no_print", "--game_name", "Glove-0.7-LA"])
        counter += 50

    # w2v lookahead
    counter = 100
    for i in range(30):
        str_counter = str(int(counter))
        subprocess.run(["python", "run_game.py", "players.codemaster_w2v_lookahead.AICodemaster", "players.guesser_w2v.AIGuesser",
                        "--seed", str_counter, "--w2v", "cache_files/cache_w2v.txt", "--no_print", "--game_name", "W2V-0.7-LA"])
        counter += 50

run()
