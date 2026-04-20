"""
Run ablation matchups in parallel, save logs, and generate plots automatically.

Usage
-----
    python3 run_stats.py mcts        # MCTS ablation only
    python3 run_stats.py minmax      # Minmax ablation only
    python3 run_stats.py heuristic   # Heuristic ablation only
    python3 run_stats.py baseline    # Final agent vs baselines
    python3 run_stats.py             # Everything

Logs are saved to:
    logs/<target>_<timestamp>/
        games.csv       — one row per game (matchup, game id, result, moves, times)
        results.json    — aggregated stats, ready to reload

Plots are generated automatically in:
    plots/
        plot_ablation.png
        table_ablation.png

Agent versions
--------------
MCTS ablation (opponent: minmax_agent.py)
    mcts_agent1  — base: no normalization, uniform priors, no tree reuse
    mcts_agent2  — + value normalization
    mcts_agent3  — + heuristic priors
    mcts_agent4  — + tree reuse (update_root)
    mcts_agent5  — + early playout termination

Minmax ablation (opponent: mcts_agent1.py)
    minmax_agent1 — base: alpha-beta + IDS
    minmax_agent2 — + move ordering
    minmax_agent3 — + killer moves
    minmax_agent4 — + transposition table (exact/lower/upper bounds)
    minmax_agent5 — + quiescence search
    minmax_agent6 — + late move reduction
    minmax_agent7 — + singular extensions
    minmax_agent8 — + probcut
    minmax_agent9 — + aspiration windows

Heuristic ablation (agent: mcts_agent4, opponent: minmax_agent.py)
    heuristic_v1  — base: line scores + immediate threats
    heuristic_v2  — + piece count advantage
    heuristic     — + totem control (full)
"""

import csv
import json
import multiprocessing
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from manager import AgentProcess
from oxono import Game, State
from stats_plot import generate_plots

# ── Configuration ──────────────────────────────────────────────────────────────

TIME_LIMIT_SECONDS = 300
NUM_GAMES          = 20
NUM_WORKERS        = 7    # 7 parallel games × 2 agent processes = 14 cores used

MCTS      = "agents/mcts/"
MM        = "agents/minmax/"
MM_TUNED  = "agents/minmax_tuned/"

MCTS_ABLATION_MATCHUPS = [
    ("MCTS base",               MCTS+"mcts_agent1.py",    "minmax_agent.py"),
    ("+ normalisation",         MCTS+"mcts_agent2.py",    "minmax_agent.py"),
    ("+ priors heuristique",    MCTS+"mcts_agent3.py",    "minmax_agent.py"),
    ("+ réutilisation arbre",   MCTS+"mcts_agent4.py",    "minmax_agent.py"),
    ("+ early playout",         MCTS+"mcts_agent5.py",    "minmax_agent.py"),
]

MINMAX_ABLATION_MATCHUPS = [
    ("Minmax base",             MM+"minmax_agent1.py",    MCTS+"mcts_agent1.py"),
    ("+ ordonnancement",        MM+"minmax_agent2.py",    MCTS+"mcts_agent1.py"),
    ("+ killer moves",          MM+"minmax_agent3.py",    MCTS+"mcts_agent1.py"),
    ("+ table transposition",   MM+"minmax_agent4.py",    MCTS+"mcts_agent1.py"),
    ("+ quiescence",            MM+"minmax_agent5.py",    MCTS+"mcts_agent1.py"),
    ("+ LMR",                   MM+"minmax_agent6.py",    MCTS+"mcts_agent1.py"),
    ("+ singular extensions",   MM+"minmax_agent7.py",    MCTS+"mcts_agent1.py"),
    ("+ probcut",               MM+"minmax_agent8.py",    MCTS+"mcts_agent1.py"),
    ("+ aspiration windows",    MM+"minmax_agent9.py",    MCTS+"mcts_agent1.py"),
]

MINMAX_VS_MCTS4_MATCHUPS = [
    ("Minmax base",             MM+"minmax_agent1.py",    MCTS+"mcts_agent4.py"),
    ("+ ordonnancement",        MM+"minmax_agent2.py",    MCTS+"mcts_agent4.py"),
    ("+ killer moves",          MM+"minmax_agent3.py",    MCTS+"mcts_agent4.py"),
    ("+ table transposition",   MM+"minmax_agent4.py",    MCTS+"mcts_agent4.py"),
    ("+ quiescence",            MM+"minmax_agent5.py",    MCTS+"mcts_agent4.py"),
    ("+ LMR",                   MM+"minmax_agent6.py",    MCTS+"mcts_agent4.py"),
    ("+ singular extensions",   MM+"minmax_agent7.py",    MCTS+"mcts_agent4.py"),
    ("+ probcut",               MM+"minmax_agent8.py",    MCTS+"mcts_agent4.py"),
    ("+ aspiration windows",    MM+"minmax_agent9.py",    MCTS+"mcts_agent4.py"),
]

HEURISTIC_ABLATION_MATCHUPS = [
    ("Heuristique base",        MCTS+"mcts_agent4_hv1.py", "minmax_agent.py"),
    ("+ pièces restantes",      MCTS+"mcts_agent4_hv2.py", "minmax_agent.py"),
    ("+ contrôle totems",       MCTS+"mcts_agent4.py",     "minmax_agent.py"),
]

MINMAX_TUNED_MATCHUPS = [
    ("+ singular (margin=2000)", MM_TUNED+"minmax_tuned2.py", MCTS+"mcts_agent4.py"),
    ("+ probcut (margin=800)",   MM_TUNED+"minmax_tuned3.py", MCTS+"mcts_agent4.py"),
    ("+ aspiration windows",     MM_TUNED+"minmax_tuned4.py", MCTS+"mcts_agent4.py"),
]

BASELINE_MATCHUPS = [
    ("vs Aléatoire",            MCTS+"mcts_agent4.py", "random_agent.py"),
    ("vs Minmax final",         MCTS+"mcts_agent4.py", MM+"minmax_agent9.py"),
    ("vs MCTS base",            MCTS+"mcts_agent4.py", MCTS+"mcts_agent1.py"),
]

HYBRID        = "agents/hybrid/"
MM_BEST       = MM_TUNED + "minmax_tuned4.py"
MCTS_BEST     = MCTS     + "mcts_agent3.py"
HYBRID_AGENT  = HYBRID   + "hybrid_agent.py"

FINAL_MATCHUPS = [
    ("Hybrid vs Minmax tuned",  HYBRID_AGENT,  MM_BEST),
    ("Hybrid vs MCTS best",     HYBRID_AGENT,  MCTS_BEST),
    ("MCTS best vs Minmax tuned", MCTS_BEST,   MM_BEST),
]

BAR_TITLES = {
    "mcts":          "Progression de l'agent MCTS — 20 parties vs Minmax",
    "minmax":        "Progression de l'agent Minmax — 20 parties vs MCTS base",
    "minmax_v4":     "Progression de l'agent Minmax — 20 parties vs MCTS v4",
    "minmax_tuned":  "Minmax features tunées — 20 parties vs MCTS v4",
    "heuristic":     "Impact de l'heuristique — 20 parties vs Minmax",
    "baseline":      "Agent final vs références — 20 parties par matchup",
    "final":         "Comparaison finale — Hybrid vs MCTS best vs Minmax tuned",
}

# ── Game runner ────────────────────────────────────────────────────────────────

def play_instrumented(agent_file_0: str, agent_file_1: str, time_limit: int,
                      swap_sides: bool = False) -> dict:
    # swap_sides: agent_file_0 plays as player 1, agent_file_1 plays as player 0.
    # Result is always reported from agent_file_0's perspective.
    if swap_sides:
        a0 = AgentProcess(agent_file_0, 1)
        a1 = AgentProcess(agent_file_1, 0)
        player_of_a0 = 1
    else:
        a0 = AgentProcess(agent_file_0, 0)
        a1 = AgentProcess(agent_file_1, 1)
        player_of_a0 = 0

    remaining_times = [float(time_limit), float(time_limit)]
    move_counts = [0, 0]

    state = State()
    result = None
    try:
        while not Game.is_terminal(state) and all(t > 0 for t in remaining_times):
            current = Game.to_move(state)
            agent = a0 if current == player_of_a0 else a1
            try:
                action, elapsed = agent.get_action(state.copy(), remaining_times[current])
                remaining_times[current] -= elapsed
                move_counts[current] += 1
            except (TimeoutError, RuntimeError):
                result = (-1, 1) if current == player_of_a0 else (1, -1)
                break
            if action not in Game.actions(state):
                result = (-1, 1) if current == player_of_a0 else (1, -1)
                break
            Game.apply(state, action)

        if result is None:
            if remaining_times[player_of_a0] <= 0:
                result = (-1, 1)
            elif remaining_times[1 - player_of_a0] <= 0:
                result = (1, -1)
            else:
                result = (Game.utility(state, player_of_a0),
                          Game.utility(state, 1 - player_of_a0))
    finally:
        a0.shutdown()
        a1.shutdown()

    return {
        "result":     result,
        "move_count": move_counts[0] + move_counts[1],
        "elapsed_p0": time_limit - remaining_times[player_of_a0],
        "elapsed_p1": time_limit - remaining_times[1 - player_of_a0],
        "moves_p0":   move_counts[player_of_a0],
        "moves_p1":   move_counts[1 - player_of_a0],
    }

# ── Aggregation ────────────────────────────────────────────────────────────────

def aggregate(game_stats: list[dict]) -> tuple:
    wins = draws = losses = 0
    total_moves = total_time_p0 = total_moves_p0 = total_duration = 0

    for s in game_stats:
        u = s["result"][0]
        if u == 1:     wins   += 1
        elif u == 0:   draws  += 1
        else:          losses += 1
        total_moves    += s["move_count"]
        total_time_p0  += s["elapsed_p0"]
        total_moves_p0 += s["moves_p0"]
        total_duration += s["elapsed_p0"] + s["elapsed_p1"]

    n = len(game_stats)
    return (
        wins, draws, losses,
        round(total_moves    / n,                     1),
        round(total_time_p0  / max(total_moves_p0, 1), 2),
        round(total_duration / n,                     1),
    )

# ── Matchup runner ─────────────────────────────────────────────────────────────

def run_matchup(label: str, agent_0: str, agent_1: str,
                csv_writer, csv_file) -> tuple:
    print(f"  [{label}] {NUM_GAMES} parties : {agent_0} vs {agent_1}")
    game_stats = []

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {
            executor.submit(play_instrumented, agent_0, agent_1, TIME_LIMIT_SECONDS,
                            swap_sides=(i >= NUM_GAMES // 2)): i
            for i in range(NUM_GAMES)
        }
        for future in as_completed(futures):
            game_id = futures[future]
            stats = future.result()
            game_stats.append(stats)

            u = stats["result"][0]
            outcome = "WIN" if u == 1 else ("NUL" if u == 0 else "PERTE")
            print(f"    partie {len(game_stats):>2}/{NUM_GAMES} → {outcome}  "
                  f"({stats['move_count']} coups, {stats['elapsed_p0'] + stats['elapsed_p1']:.0f}s)")

            csv_writer.writerow([
                label, game_id, outcome,
                stats["move_count"],
                round(stats["elapsed_p0"], 2),
                round(stats["elapsed_p1"], 2),
                round(stats["elapsed_p0"] + stats["elapsed_p1"], 2),
            ])
            csv_file.flush()

    agg = aggregate(game_stats)
    print(f"  [{label}] V={agg[0]} N={agg[1]} D={agg[2]}  "
          f"({agg[3]} coups moy., {agg[5]}s/partie)\n")
    return (label, *agg)

# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)

    target = sys.argv[1] if len(sys.argv) > 1 else "all"

    matchup_map = {
        "mcts":          MCTS_ABLATION_MATCHUPS,
        "minmax":        MINMAX_ABLATION_MATCHUPS,
        "minmax_v4":     MINMAX_VS_MCTS4_MATCHUPS,
        "minmax_tuned":  MINMAX_TUNED_MATCHUPS,
        "heuristic":     HEURISTIC_ABLATION_MATCHUPS,
        "baseline":      BASELINE_MATCHUPS,
        "final":         FINAL_MATCHUPS,
    }
    targets = list(matchup_map.keys()) if target == "all" else [target]

    for t in targets:
        matchups = matchup_map[t]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = Path("logs") / f"{t}_{timestamp}"
        log_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"  {t.upper()} ABLATION  —  logs → {log_dir}")
        print(f"{'='*60}\n")

        results = []
        csv_path = log_dir / "games.csv"
        with open(csv_path, "w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["matchup", "game_id", "outcome", "moves",
                             "elapsed_p0", "elapsed_p1", "duration"])
            for matchup in matchups:
                entry = run_matchup(*matchup, writer, csv_file)
                results.append(entry)

        # Save aggregated results
        results_path = log_dir / "results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Résultats sauvegardés → {results_path}")

        # Generate plots automatically into plots/<target>/
        generate_plots(results, BAR_TITLES[t], output_dir=f"plots/{t}")
