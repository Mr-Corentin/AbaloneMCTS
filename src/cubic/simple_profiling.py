import time
from typing import Any, Dict
import jax
from jax import profiler
import jax.numpy as jnp
from env import AbaloneEnv, AbaloneState
import mctx
from mcts import run_search, AbaloneMCTSRecurrentFn, get_root_output

class SimpleProfiler:
    def __init__(self):
        self.timings: Dict[str, float] = {}
        self.counts: Dict[str, int] = {}
        self.start_times: Dict[str, float] = {}

    def start(self, name: str):
        """Démarre le timing pour une section"""
        self.start_times[name] = time.perf_counter()

    def stop(self, name: str):
        """Arrête le timing et enregistre la durée"""
        if name in self.start_times:
            duration = time.perf_counter() - self.start_times[name]
            self.timings[name] = self.timings.get(name, 0) + duration
            self.counts[name] = self.counts.get(name, 0) + 1

    def print_stats(self):
        """Affiche les statistiques"""
        print("\n=== Statistiques de profiling ===")
        for name in self.timings:
            total_time = self.timings[name]
            count = self.counts[name]
            avg_time = total_time / count
            print(f"\n{name}:")
            print(f"  Temps total: {total_time:.4f}s")
            print(f"  Appels: {count}")
            print(f"  Temps moyen: {avg_time:.4f}s")
def profile_run_search(state, recurrent_fn, network, params, rng_key, env, num_simulations=100, profiler=None):
    """Version profilée de run_search"""
    if profiler:
        profiler.start("get_root_output")
        root = get_root_output(state, network, params, env)
        profiler.stop("get_root_output")
        
        profiler.start("get_legal_moves")
        invalid_actions = jnp.expand_dims(~env.get_legal_moves(state), 0)
        profiler.stop("get_legal_moves")
        
        profiler.start("muzero_policy")
        policy_output = mctx.gumbel_muzero_policy(
            params=params,
            rng_key=rng_key,
            root=root,
            recurrent_fn=recurrent_fn.recurrent_fn,
            num_simulations=num_simulations,
            max_num_considered_actions=60,
            invalid_actions=invalid_actions,
            gumbel_scale=0.0
        )
        print(f"\nStatistiques de l'arbre MCTS:")
        tree = policy_output.search_tree
        print(f"Nombre de nœuds dans l'arbre: {tree.node_visits.shape[0]}")
        print(f"Nombre total de visites: {jnp.sum(tree.node_visits)}")
        print(f"Visites par nœud: {tree.node_visits}")
        profiler.stop("muzero_policy")
        return policy_output
    else:
        return run_search(state, recurrent_fn, network, params, rng_key, env, num_simulations)
def profile_game(env, model, params, num_simulations=10, max_moves=5):
    profiler = SimpleProfiler()
    state = env.reset()
    move_count = 0

    while not env.is_terminal(state) and move_count < max_moves:
        move_count += 1
        print(f"\n=== Coup n°{move_count} sur {max_moves} ===")

        policy_output = profile_run_search(
            state=state,
            recurrent_fn=AbaloneMCTSRecurrentFn(env, model),
            network=model,
            params=params,
            rng_key=jax.random.PRNGKey(0),
            env=env,
            num_simulations=num_simulations,
            profiler=profiler
        )
        print("Recherche MCTS terminée, attente du ready...")
        policy_output.action.block_until_ready()
        print("Action ready terminé")
        profiler.stop("mcts_search")

        print("\nAffichage de l'état actuel...")
        print(f"Billes noires sorties : {state.black_out}")
        print(f"Billes blanches sorties : {state.white_out}")
        print(f"Joueur actuel : {state.actual_player}")
        
        print("\nAffichage des statistiques MCTS...")
        print(f"Action selected: {policy_output.action}")

        print("\nExécution du mouvement...")
        profiler.start("move_execution")
        action = policy_output.action
        print("Action récupérée, exécution du step...")
        state = env.step(state, action)
        print("Step effectué, attente du ready...")
        state.board.block_until_ready()
        print("Board ready terminé")
        profiler.stop("move_execution")

    print("\nPartie terminée, affichage des stats...")
    profiler.print_stats()


if __name__ == "__main__":
    import jax.random as random
    from network import DummyAbaloneModel

    # Initialisation
    env = AbaloneEnv()
    dummy_model = DummyAbaloneModel()
    rng_key = random.PRNGKey(0)
    
    # Initialisation des paramètres
    dummy_board = jnp.zeros((1, 9, 9))
    dummy_marbles = jnp.zeros((1, 2))
    params = dummy_model.init(rng_key, dummy_board, dummy_marbles)

    # Test avec différents nombres de simulations
    for num_sims in [5, 50, 200, 300]:
        print(f"\n\n=== Test avec {num_sims} simulations ===")
        print("=" * 40)
        profile_game(env, dummy_model, params, num_simulations=num_sims, max_moves=5)


# if __name__ == "__main__":
#     import jax.random as random
#     from network import DummyAbaloneModel

#     # Initialisation
#     env = AbaloneEnv()
#     dummy_model = DummyAbaloneModel()
#     rng_key = random.PRNGKey(0)
    
#     # Initialisation des paramètres
#     dummy_board = jnp.zeros((1, 9, 9))
#     dummy_marbles = jnp.zeros((1, 2))
#     params = dummy_model.init(rng_key, dummy_board, dummy_marbles)

#     # Lancer le profiling
#     profile_game(env, dummy_model, params, num_simulations=10)

# Le reste du code (main) reste inchangé