
import mctx
import jax
import jax.numpy as jnp
import chex
from typing import Tuple
from functools import partial
from env import AbaloneEnv, AbaloneState
from network import AbaloneModel, prepare_input

from time import time

@partial(jax.jit)
def calculate_reward(current_state: AbaloneState, next_state: AbaloneState) -> float:
    """
    Calcule la récompense d'une transition en version canonique
    """
    black_diff = next_state.black_out - current_state.black_out
    white_diff = next_state.white_out - current_state.white_out
    
    billes_sorties = jnp.where(current_state.actual_player == 1,
                              white_diff,
                              black_diff)
    
    return billes_sorties

@partial(jax.jit)
def calculate_discount(state: AbaloneState) -> float:
    """
    Retourne le facteur d'atténuation en utilisant jnp.where
    """
    is_terminal = (state.black_out >= 6) | (state.white_out >= 6) | (state.moves_count >= 300)
    return jnp.where(is_terminal, 0.0, 1.0)



# class AbaloneMCTSRecurrentFn:
#     def __init__(self, env: AbaloneEnv, network: AbaloneModel):
#         self.env = env
#         self.network = network

#     def recurrent_fn(self, params, rng_key, action, embedding):
#         t_start = time()
        
#         # Reconstruction de l'état
#         current_state = AbaloneState(
#             board=embedding['board_3d'][0],
#             actual_player=embedding['actual_player'][0],
#             black_out=embedding['black_out'][0],
#             white_out=embedding['white_out'][0],
#             moves_count=embedding['moves_count'][0]
#         )
#         t_reconstruct = time()
#         print(f"  Temps reconstruction état: {t_reconstruct - t_start:.3f}s")
        
#         # Prochain état
#         next_state = self.env.step(current_state, action)
#         t_step = time()
#         print(f"  Temps step: {t_step - t_reconstruct:.3f}s")
        
#         # Reward et discount
#         reward = calculate_reward(current_state, next_state)
#         reward = jnp.expand_dims(reward, 0)
#         discount = calculate_discount(next_state)
#         discount = jnp.expand_dims(discount, 0)
#         t_reward = time()
#         print(f"  Temps reward/discount: {t_reward - t_step:.3f}s")
        
#         # Préparation pour le réseau
#         actual_player = next_state.actual_player.reshape(())
#         our_marbles = jnp.where(actual_player == 1,
#                             next_state.black_out,
#                             next_state.white_out)
#         opp_marbles = jnp.where(actual_player == 1,
#                             next_state.white_out,
#                             next_state.black_out)
#         board_2d, marbles_out = prepare_input(next_state.board, our_marbles, opp_marbles)
#         t_prep = time()
#         print(f"  Temps préparation réseau: {t_prep - t_reward:.3f}s")
        
#         # Forward pass
#         prior_logits, value = self.network.apply(params, board_2d, marbles_out)
#         t_network = time()
#         print(f"  Temps réseau: {t_network - t_prep:.3f}s")
#         legal_moves = self.env.get_legal_moves(next_state)
#         t_legal_mpve = time()
#         print(f"  Temps legal moves: {t_legal_mpve - t_network:.3f}s")
#         prior_logits = jnp.where(legal_moves, 
#                                 prior_logits, 
#                                 jnp.full_like(prior_logits, float('-inf')))
#         # Créer le nouvel embedding avec dimensions de batch
#         next_embedding = {
#             'board_3d': jnp.expand_dims(next_state.board, 0),
#             'board_2d': board_2d,
#             'actual_player': jnp.expand_dims(next_state.actual_player, 0),
#             'black_out': jnp.expand_dims(next_state.black_out, 0),
#             'white_out': jnp.expand_dims(next_state.white_out, 0),
#             'moves_count': jnp.expand_dims(next_state.moves_count, 0)
#         }
        
#         return mctx.RecurrentFnOutput(
#             reward=reward,  # Maintenant a shape [batch_size]
#             discount=discount,  # Maintenant a shape [batch_size]
#             prior_logits=prior_logits,
#             value=value
#         ), next_embedding

class AbaloneMCTSRecurrentFn:
    def __init__(self, env: AbaloneEnv, network: AbaloneModel):
        self.env = env
        self.network = network

    @partial(jax.jit, static_argnums=(0,))
    def recurrent_fn(self, params, rng_key, action, embedding):
        # Version sans prints et optimisée pour JAX
        current_state = AbaloneState(
            board=embedding['board_3d'][0],
            actual_player=embedding['actual_player'][0],
            black_out=embedding['black_out'][0],
            white_out=embedding['white_out'][0],
            moves_count=embedding['moves_count'][0]
        )
        
        next_state = self.env.step(current_state, action)
        
        # Utiliser des opérations vectorisées quand possible
        reward = calculate_reward(current_state, next_state)
        reward = jnp.expand_dims(reward, 0)
        discount = calculate_discount(next_state)
        discount = jnp.expand_dims(discount, 0)
        
        # Regrouper les calculs pour éviter les opérations intermédiaires
        actual_player = next_state.actual_player.reshape(())
        marbles = jnp.where(actual_player == 1,
                           jnp.array([next_state.black_out, next_state.white_out]),
                           jnp.array([next_state.white_out, next_state.black_out]))
        
        board_2d, marbles_out = prepare_input(next_state.board, marbles[0], marbles[1])
        prior_logits, value = self.network.apply(params, board_2d, marbles_out)
        
        legal_moves = self.env.get_legal_moves(next_state)
        prior_logits = jnp.where(legal_moves, prior_logits, jnp.full_like(prior_logits, float('-inf')))
        
        next_embedding = {
            'board_3d': jnp.expand_dims(next_state.board, 0),
            'actual_player': jnp.expand_dims(next_state.actual_player, 0),
            'black_out': jnp.expand_dims(next_state.black_out, 0),
            'white_out': jnp.expand_dims(next_state.white_out, 0),
            'moves_count': jnp.expand_dims(next_state.moves_count, 0)
        }
        
        return mctx.RecurrentFnOutput(reward, discount, prior_logits, value), next_embedding

@partial(jax.jit, static_argnames=['network', 'env', 'batch_size'])
def get_root_output(state: AbaloneState, network: AbaloneModel, params, env: AbaloneEnv, batch_size: int = 1):
    our_marbles = jnp.where(state.actual_player == 1,
                           state.black_out,
                           state.white_out)
    opp_marbles = jnp.where(state.actual_player == 1,
                           state.white_out,
                           state.black_out)
    
    board_2d, marbles_out = prepare_input(state.board, our_marbles, opp_marbles)
    prior_logits, value = network.apply(params, board_2d, marbles_out)
    
    embedding = {
        'board_3d': jnp.expand_dims(state.board, 0),
        'board_2d': board_2d,
        'actual_player': jnp.array([state.actual_player]),
        'black_out': jnp.array([state.black_out]),
        'white_out': jnp.array([state.white_out]),
        'moves_count': jnp.array([state.moves_count])
    }
    
    return mctx.RootFnOutput(
        prior_logits=prior_logits,
        value=value,
        embedding=embedding
    )

#good one::
@partial(jax.jit, static_argnames=['recurrent_fn', 'network', 'env', 'num_simulations', 'max_num_considered_actions'])
def run_search(state: AbaloneState, 
              recurrent_fn: AbaloneMCTSRecurrentFn, 
              network: AbaloneModel,
              params,
              rng_key,
              env: AbaloneEnv,
              num_simulations: int = 100,
              max_num_considered_actions: int = 60):
    
    root = get_root_output(state, network, params, env)
    
    # Obtenir le masque et ajuster ses dimensions
    legal_moves = env.get_legal_moves(state)
    invalid_actions = ~legal_moves
    
    # Ajouter la dimension de batch pour correspondre à prior_logits
    invalid_actions = jnp.expand_dims(invalid_actions, 0)  # Shape devient (1, 1734)
    
    policy_output = mctx.gumbel_muzero_policy(
        params=params,
        rng_key=rng_key,
        root=root,
        recurrent_fn=recurrent_fn.recurrent_fn,
        num_simulations=num_simulations,
        max_num_considered_actions=max_num_considered_actions,
        invalid_actions=invalid_actions,
        gumbel_scale=0.0
    )
    
    return policy_output



def test_root_output():
    """Test de la création du RootFnOutput"""
    print("Test de RootFnOutput :")
    
    # Initialiser l'environnement et le réseau
    env = AbaloneEnv()
    network = AbaloneModel()
    
    # Créer une clé pour la génération aléatoire
    rng = jax.random.PRNGKey(0)
    
    # Initialiser l'état
    state = env.reset()
    
    # Debug: vérifier l'entrée du réseau
    board_2d, marbles_out = prepare_input(state.board, 0, 0)
    print(f"\nBoard shape: {board_2d.shape}")
    print(f"Marbles shape: {marbles_out.shape}")
    print(f"Board contains NaN: {jnp.any(jnp.isnan(board_2d))}")
    print(f"Marbles contains NaN: {jnp.any(jnp.isnan(marbles_out))}")
    
    # Initialiser le réseau
    variables = network.init(rng, board_2d, marbles_out)
    
    # Debug: test direct du réseau
    prior_logits, value = network.apply(variables, board_2d, marbles_out)
    print(f"\nTest direct du réseau:")
    print(f"Prior_logits shape: {prior_logits.shape}")
    print(f"Value shape: {value.shape}")
    print(f"Prior_logits contains NaN: {jnp.any(jnp.isnan(prior_logits))}")
    print(f"Value contains NaN: {jnp.any(jnp.isnan(value))}")
    
    # Créer le RootFnOutput
    root = get_root_output(state, network, variables)
    
    # Vérifier les shapes et les valeurs
    print(f"\nRootFnOutput final:")
    print(f"Shape des prior_logits : {root.prior_logits.shape}")
    print(f"Shape de la value : {root.value.shape}")
    print(f"Type de l'embedding : {type(root.embedding)}")
    print(f"Value range : [{float(jnp.min(root.value))}, {float(jnp.max(root.value))}]")
    print(f"Prior_logits range : [{float(jnp.min(root.prior_logits))}, {float(jnp.max(root.prior_logits))}]")


def test_recurrent_fn():
    """Test de la fonction récurrente pour MCTS"""
    print("\nTest de RecurrentFn:")
    
    # Initialiser l'environnement et le réseau
    env = AbaloneEnv()
    network = AbaloneModel()
    recurrent_fn = AbaloneMCTSRecurrentFn(env, network)
    
    # Créer une clé pour la génération aléatoire
    rng = jax.random.PRNGKey(0)
    
    # Initialiser l'état
    state = env.reset()
    
    # Initialiser le réseau
    board_2d, marbles_out = prepare_input(state.board, 0, 0)
    variables = network.init(rng, board_2d, marbles_out)
    # Vérifier les shapes des entrées du réseau
    print(f"Shape board_2d: {board_2d.shape}")
    print(f"Shape marbles_out: {marbles_out.shape}")

  
        # Obtenir les coups légaux
    legal_moves = env.get_legal_moves(state)
    first_legal_move = jnp.argmax(legal_moves)  # Premier coup légal
    
    # Tester la fonction récurrente
    print(f"\nTest avec l'action {first_legal_move}:")
    output, next_state = recurrent_fn.recurrent_fn(variables, rng, first_legal_move, state)
    
    print("\nSortie de RecurrentFn:")
    print(f"Shape prior_logits: {output.prior_logits.shape}")
    print(f"Shape value: {output.value.shape}")
    print(f"Reward: {output.reward}")
    print(f"Discount: {output.discount}")
    
    print("\nVérification du prochain état:")
    print(f"Billes noires sorties: {next_state.black_out}")
    print(f"Billes blanches sorties: {next_state.white_out}")
    print(f"Joueur actuel: {next_state.actual_player}")
def test_search():
    """Test de la recherche MCTS avec Gumbel MuZero"""
    print("\nTest de la recherche MCTS avec Gumbel MuZero:")
    
    # Initialiser tout
    env = AbaloneEnv()
    network = AbaloneModel()
    recurrent_fn = AbaloneMCTSRecurrentFn(env, network)
    
    # Créer les clés aléatoires
    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)
    
    # Initialiser l'état et le réseau
    state = env.reset()
    board_2d, marbles_out = prepare_input(state.board, 0, 0)
    variables = network.init(rng, board_2d, marbles_out)
    
    print("\nVérification de l'état initial:")
    print(f"Joueur actuel: {state.actual_player}")
    print(f"Billes noires sorties: {state.black_out}")
    print(f"Billes blanches sorties: {state.white_out}")
    
    # Lancer la recherche
    print("\nDébug moves_index shapes:")
    print(f"positions shape: {env.moves_index['positions'].shape}")
    print(f"directions shape: {env.moves_index['directions'].shape}")
    print(f"move_types shape: {env.moves_index['move_types'].shape}")
    print(f"group_sizes shape: {env.moves_index['group_sizes'].shape}")
    print("\nLancement de la recherche...")
    policy_output = run_search(
        state=state,
        recurrent_fn=recurrent_fn,
        network=network,
        params=variables,
        rng_key=rng,
        env=env
    )
    
    print("\nRésultats de la recherche:")
    print(f"Action choisie: {policy_output.action}")
    print(f"Shape des poids des actions: {policy_output.action_weights.shape}")
    print("\nAnalyse de l'action choisie:")
    # Vérifier si l'action est légale
    legal_moves = env.get_legal_moves(state)
    chosen_action = policy_output.action[0]  # Enlever la dimension de batch
    print(f"L'action est légale : {legal_moves[chosen_action]}")
    
    # Récupérer les top k actions
    top_k = 5
    action_weights = policy_output.action_weights[0]  # Enlever la dimension de batch
    top_indices = jnp.argsort(action_weights)[-top_k:][::-1]
    print(f"\nTop {top_k} actions :")
    for idx in top_indices:
        print(f"Action {idx}: probabilité {action_weights[idx]:.3f}")
    
if __name__ == "__main__":
    test_search()
# if __name__ == "__main__":
#     test_recurrent_fn()
# if __name__ == "__main__":
#     test_root_output()
# if __name__ == "__main__":
#     test_mcts_search()
    
# def test_mcts():
#     """Test de l'intégration MCTS"""
#     # Initialiser l'environnement et le réseau
#     env = AbaloneEnv()
#     network = AbaloneModel()
#     recurrent_fn = AbaloneMCTSRecurrentFn(env, network)
    
#     # Créer une clé pour la génération aléatoire
#     rng = jax.random.PRNGKey(0)
#     rng, init_rng = jax.random.split(rng)
    
#     # Initialiser l'état et les paramètres du réseau
#     state = recurrent_fn.init(init_rng)
    
#     # Créer des données de test pour le réseau
#     board_2d, marbles_out = prepare_input(state.board, 0, 0)
#     variables = network.init(rng, board_2d, marbles_out)
    
#     # Tester un pas de MCTS
#     rng, action_rng = jax.random.split(rng)
#     action = 0  # Premier coup possible
    
#     # Appliquer l'action
#     next_state, prior, value = recurrent_fn.apply(variables, state, action, action_rng)
    
#     print("Test MCTS :")
#     print(f"Forme du prior : {prior.shape}")
#     print(f"Forme de la valeur : {value.shape}")
#     print(f"État terminal ? : {recurrent_fn.is_terminal(next_state)}")
#     legal_moves = recurrent_fn.get_legal_moves(next_state)
#     print(f"Nombre de coups légaux : {jnp.sum(legal_moves)}")

# if __name__ == "__main__":
#     test_mcts()