import time
import jax
import jax.numpy as jnp
from parallel_mcts import parallel_dummy_search
from typing import Dict

class SimpleProfiler:
    def __init__(self):
        self.timings: Dict[str, float] = {}
        self.counts: Dict[str, int] = {}
        self.start_times: Dict[str, float] = {}

    def start(self, name: str):
        self.start_times[name] = time.perf_counter()

    def stop(self, name: str):
        if name in self.start_times:
            duration = time.perf_counter() - self.start_times[name]
            self.timings[name] = self.timings.get(name, 0) + duration
            self.counts[name] = self.counts.get(name, 0) + 1

    def print_stats(self):
        print("\n=== Statistiques de profiling ===")
        for name in self.timings:
            total_time = self.timings[name]
            count = self.counts[name]
            avg_time = total_time / count
            print(f"\n{name}:")
            print(f"  Temps total: {total_time:.4f}s")
            print(f"  Appels: {count}")
            print(f"  Temps moyen: {avg_time:.4f}s")

def profile_game_parallel(env, dummy_model, params, num_simulations=300, max_moves=5):
    profiler = SimpleProfiler()
    state = env.reset()
    move_count = 0

    while not env.is_terminal(state) and move_count < max_moves:
        move_count += 1
        print(f"\n=== Coup n°{move_count} sur {max_moves} ===")

        profiler.start("parallel_mcts_search")
        policy_outputs = parallel_dummy_search(
            state,
            dummy_model,
            params,
            jax.random.PRNGKey(move_count),
            env,
            num_simulations
        )
        
        action = jax.device_get(policy_outputs.action)[0]
        profiler.stop("parallel_mcts_search")

        print("\nAffichage de l'état actuel...")
        print(f"Billes noires sorties : {state.black_out}")
        print(f"Billes blanches sorties : {state.white_out}")
        print(f"Joueur actuel : {state.actual_player}")
        
        print("\nAffichage des statistiques MCTS...")
        print(f"Action selected: {action}")

        print("\nExécution du mouvement...")
        profiler.start("move_execution")
        state = env.step(state, action)
        state.board.block_until_ready()
        profiler.stop("move_execution")

    print("\nPartie terminée, affichage des stats...")
    profiler.print_stats()