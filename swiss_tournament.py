import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from run_stats import aggregate, play_instrumented
from stats_plot import generate_plots

# Configuration du Tournoi
TIME_LIMIT_SECONDS = 300
NUM_WORKERS = 4
MATCHES_PER_SIDE = 5

AGENTS = [
    "athenan_agent.py",
    "mcts_agent.py",
    "minmax_agent.py",
    "random_agent.py",
]

NUM_ROUNDS = math.ceil(math.log2(len(AGENTS))) + 1


def generate_pairings(agents_ranked, past_matches):
    pairings = []
    unpaired = list(agents_ranked)

    while unpaired:
        p1 = unpaired.pop(0)
        paired_successfully = False

        for i, p2 in enumerate(unpaired):
            match_set = frozenset([p1, p2])
            if match_set not in past_matches:
                pairings.append((p1, p2))
                unpaired.pop(i)
                past_matches.add(match_set)
                paired_successfully = True
                break

        if not paired_successfully and unpaired:
            p2 = unpaired.pop(0)
            pairings.append((p1, p2))

    return pairings


def run_tournament():
    print(
        f"Swiss Tournament - {NUM_ROUNDS} Rounds, {MATCHES_PER_SIDE * 2} Matches per Pairing"
    )
    print("=" * 60)

    scores = {agent: 0 for agent in AGENTS}
    past_matches = set()
    all_game_stats = {agent: [] for agent in AGENTS}

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        for round_num in range(1, NUM_ROUNDS + 1):
            print(f"Round {round_num}/{NUM_ROUNDS}")

            ranked_agents = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
            pairings = generate_pairings(ranked_agents, past_matches)

            futures = {}
            for match_id, (a0, a1) in enumerate(pairings):
                print(
                    f"Match {match_id + 1}/{len(pairings)} : {Path(a0).stem} vs {Path(a1).stem} ( score: {scores[a0]} - {scores[a1]} )"
                )

                for _ in range(MATCHES_PER_SIDE):
                    f1 = executor.submit(
                        play_instrumented, a0, a1, TIME_LIMIT_SECONDS, swap_sides=False
                    )
                    futures[f1] = (a0, a1)

                    f2 = executor.submit(
                        play_instrumented, a0, a1, TIME_LIMIT_SECONDS, swap_sides=True
                    )
                    futures[f2] = (a0, a1)

            completed_matches = 0
            total_matches_in_round = len(pairings) * MATCHES_PER_SIDE * 2

            for future in as_completed(futures):
                a0, a1 = futures[future]
                stats = future.result()

                u0, u1 = stats["result"]
                if u0 == 1:
                    scores[a0] += 3
                elif u1 == 1:
                    scores[a1] += 3
                else:
                    scores[a0] += 1
                    scores[a1] += 1

                stats_a0 = stats.copy()
                stats_a1 = stats.copy()
                stats_a1["result"] = (u1, u0)
                stats_a1["elapsed_p0"], stats_a1["elapsed_p1"] = (
                    stats["elapsed_p1"],
                    stats["elapsed_p0"],
                )
                stats_a1["moves_p0"], stats_a1["moves_p1"] = (
                    stats["moves_p1"],
                    stats["moves_p0"],
                )

                all_game_stats[a0].append(stats_a0)
                all_game_stats[a1].append(stats_a1)

                completed_matches += 1
                if completed_matches % (MATCHES_PER_SIDE * 2) == 0:
                    print(
                        f"Progression Round {round_num}: {completed_matches}/{total_matches_in_round} matches completed"
                    )

    print("\n" + "=" * 60)
    print("Final Scores:")
    print("=" * 60)

    final_ranking = sorted(scores.items(), key=lambda item: item[1], reverse=True)

    for rank, (agent, score) in enumerate(final_ranking, 1):
        print(f"{rank}. {Path(agent).stem:<20} : {score:>3} points")

    formatted_results = []
    for agent, _ in final_ranking:
        label = Path(agent).stem
        agg_stats = aggregate(all_game_stats[agent])
        formatted_results.append((label, *agg_stats))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = f"plots/tournament_{timestamp}"

    print(f"Generating plots in {out_dir}...")
    generate_plots(
        formatted_results,
        f"Résultats Tournoi Suisse, {NUM_ROUNDS} rondes, {MATCHES_PER_SIDE * 2} matchs par rencontre",
        output_dir=out_dir,
    )


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.set_start_method("spawn", force=True)
    run_tournament()
