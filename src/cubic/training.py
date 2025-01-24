import jax
import jax.numpy as jnp
from typing import List, Tuple, Dict
import chex
from env import AbaloneEnv, AbaloneState
from network import AbaloneModel, prepare_input
from mcts import AbaloneMCTSRecurrentFn, run_search

class AbaloneReplayBuffer:
    """Buffer pour stocker les expériences de jeu"""
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.states: List[AbaloneState] = []
        self.policies: List[jnp.ndarray] = []  # politiques MCTS (π)
        self.values: List[float] = []  # résultats des parties

    def add_experience(self, state: AbaloneState, policy: jnp.ndarray, value: float):
        if len(self.states) >= self.max_size:
            # Si buffer plein, on enlève le plus ancien
            self.states.pop(0)
            self.policies.pop(0)
            self.values.pop(0)
        
        self.states.append(state)
        self.policies.append(policy)
        self.values.append(value)

# def play_game(env: AbaloneEnv, 
#               network: AbaloneModel, 
#               params,
#               recurrent_fn: AbaloneMCTSRecurrentFn,
#               rng_key: chex.PRNGKey,
#               temperature: float = 1.0) -> Tuple[List[AbaloneState], List[jnp.ndarray], float]:
#     """
#     Joue une partie complète et retourne les états, politiques et le résultat
    
#     Args:
#         env: L'environnement de jeu
#         network: Le réseau de neurones
#         params: Les paramètres du réseau
#         recurrent_fn: La fonction récurrente pour MCTS
#         rng_key: Clé pour la génération de nombres aléatoires
#         temperature: Température pour l'exploration (1.0 = beaucoup d'exploration)
    
#     Returns:
#         states: Liste des états visités
#         policies: Liste des politiques MCTS
#         result: Résultat de la partie (-1, 0, ou 1)
#     """
#     states = []
#     policies = []
#     state = env.reset()
    
#     while not env.is_terminal(state):
#         # Obtenir la politique MCTS
#         policy_output = run_search(
#             state=state,
#             recurrent_fn=recurrent_fn,
#             network=network,
#             params=params,
#             rng_key=rng_key,
#             env=env
#         )
        
#         # Sauvegarder l'état et la politique
#         states.append(state)
#         policies.append(policy_output.action_weights[0])  # Enlever dim de batch
        
#         # Faire le mouvement
#         action = policy_output.action[0]  # Enlever dim de batch
#         state = env.step(state, action)
        
#         # Mise à jour de la clé rng
#         rng_key, _ = jax.random.split(rng_key)
    
#     result = env.get_winner(state)
#     return states, policies, result

from time import time

def play_game(env: AbaloneEnv, 
              network: AbaloneModel, 
              params,
              recurrent_fn: AbaloneMCTSRecurrentFn,
              rng_key: chex.PRNGKey,
              temperature: float = 1.0):
    states = []
    policies = []
    state = env.reset()
    
    move_count = 0
    total_time = 0
    total_search_time = 0
    
    print("\nDébut de la partie avec profiling...")
    
    while not env.is_terminal(state):
        move_start = time()
        
        # Timer pour MCTS
        search_start = time()
        policy_output = run_search(
            state=state,
            recurrent_fn=recurrent_fn,
            network=network,
            params=params,
            rng_key=rng_key,
            env=env
        )
        search_time = time() - search_start
        total_search_time += search_time
        
        states.append(state)
        policies.append(policy_output.action_weights[0])
        
        # Faire le mouvement
        action = policy_output.action[0]
        state = env.step(state, action)
        
        move_time = time() - move_start
        total_time += move_time
        move_count += 1
        
        # Log tous les 5 coups
        if move_count % 5 == 0:
            print(f"Coup {move_count}:")
            print(f"  Temps moyen par coup: {total_time/move_count:.3f}s")
            print(f"  Temps moyen MCTS: {total_search_time/move_count:.3f}s")
        
        rng_key, _ = jax.random.split(rng_key)
    
    print("\nStatistiques finales:")
    print(f"Nombre total de coups: {move_count}")
    print(f"Temps total: {total_time:.2f}s")
    print(f"Temps moyen par coup: {total_time/move_count:.3f}s")
    print(f"Temps total MCTS: {total_search_time:.2f}s ({(total_search_time/total_time)*100:.1f}%)")
    
    result = env.get_winner(state)
    return states, policies, result
from network import DummyAbaloneModel

def test_self_play():
    """Test de la fonction de self-play"""
    print("\nTest de self-play:")
    
    # Initialisation
    env = AbaloneEnv()
    network = DummyAbaloneModel()
    recurrent_fn = AbaloneMCTSRecurrentFn(env, network)
    
    # Clés aléatoires
    rng = jax.random.PRNGKey(0)
    rng, init_rng, init_rng2 = jax.random.split(rng, 3)  # Split en 3
    
    # Initialiser le réseau correctement
    dummy_state = env.reset()
    dummy_board_2d, dummy_marbles_out = prepare_input(dummy_state.board, 0, 0)
    
    # # Initialiser le réseau proprement avec des shapes corrects
    # params = network.init(init_rng, 
    #                      dummy_board_2d,  # shape (1, 9, 9)
    #                      dummy_marbles_out)  # shape (1, 2)
    
    # print("\nDébug initialisation réseau:")
    # print(f"Shape dummy_board_2d: {dummy_board_2d.shape}")
    # print(f"Shape dummy_marbles_out: {dummy_marbles_out.shape}")
    
    # # Vérifier que les paramètres sont bien initialisés
    # if 'params' not in params:
    #     raise ValueError("Les paramètres n'ont pas été correctement initialisés")
    network = DummyAbaloneModel(num_actions=1734)
    recurrent_fn = AbaloneMCTSRecurrentFn(env, network)
    
    # Clés aléatoires
    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)
    
    # Initialiser le réseau
    state = env.reset()
    board_2d, marbles_out = prepare_input(state.board, 0, 0)
    params = network.init(init_rng, board_2d, marbles_out)
    
    print("Démarrage d'une partie...")
    states, policies, result = play_game(
        env=env,
        network=network,
        params=params,
        recurrent_fn=recurrent_fn,
        rng_key=rng,
        temperature=1.0
    )
    
    print(f"\nRésultats:")
    print(f"Nombre de coups joués: {len(states)}")
    print(f"Shape d'une politique: {policies[0].shape}")
    print(f"Résultat de la partie: {result}")
    
    # Test du buffer
    print("\nTest du buffer:")
    buffer = AbaloneReplayBuffer(max_size=1000)
    for state, policy in zip(states, policies):
        buffer.add_experience(state, policy, result)
    
    print(f"Taille du buffer: {len(buffer.states)}")
from time import time
import mctx
from mcts import get_root_output
from network import DummyAbaloneModel
def benchmark_mcts_config():
    print("\nBenchmark des configurations MCTS:")
    
    # Initialisation
    print("Initialisation...")
    t_start = time()
    env = AbaloneEnv()
    network = DummyAbaloneModel(num_actions=1734)
    recurrent_fn = AbaloneMCTSRecurrentFn(env, network)
    
    # Clés aléatoires
    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)
    
    # Initialiser le réseau
    state = env.reset()
    board_2d, marbles_out = prepare_input(state.board, 0, 0)
    params = network.init(init_rng, board_2d, marbles_out)
    print(f"Temps d'initialisation: {time() - t_start:.2f}s")
    
    # Configurations à tester (valeurs réduites)
    configs = [
        {"num_sims": 15},
        {"num_sims": 30},
        {"num_sims": 80}
    ]
    
    for config in configs:
        print(f"\nTest avec {config['num_sims']} simulations:")
        times = []
        
        # Un seul test pour commencer
        print("Démarrage d'un coup MCTS...")
        t_start_move = time()
        
        # Ajouter des points de mesure dans run_search
        t_root = time()
        root = get_root_output(state, network, params, env)
        t_after_root = time()
        print(f"  Temps get_root_output: {t_after_root - t_root:.2f}s")
        
        policy_output = mctx.gumbel_muzero_policy(
            params=params,
            rng_key=rng,
            root=root,
            recurrent_fn=recurrent_fn.recurrent_fn,
            num_simulations=config['num_sims'],
            max_num_considered_actions = 100
        )
        t_end = time()
        
        print(f"  Temps gumbel_muzero_policy: {t_end - t_after_root:.2f}s")
        print(f"  Temps total pour un coup: {t_end - t_start_move:.2f}s")
        
        # Vérifions aussi la validité du coup
        action = policy_output.action[0]
        legal_moves = env.get_legal_moves(state)
        print(f"  L'action choisie {action} est légale : {legal_moves[action]}")

# if __name__ == "__main__":
#     benchmark_mcts_config()

if __name__ == "__main__":
    test_self_play()