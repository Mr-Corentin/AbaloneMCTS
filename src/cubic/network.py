import flax.linen as nn
import jax
import jax.numpy as jnp
from functools import partial
from typing import Tuple
from coord_conversion import cube_to_2d  # Pour convertir notre plateau

class ResBlock(nn.Module):
    """Bloc résiduel"""
    filters: int

    @nn.compact
    def __call__(self, x):
        y = nn.Conv(self.filters, (3, 3), padding='SAME')(x)
        y = nn.relu(y)
        y = nn.Conv(self.filters, (3, 3), padding='SAME')(y)
        return nn.relu(x + y)
    
class AbaloneModel(nn.Module):
    """Réseau de neurones pour Abalone style AlphaZero"""
    num_actions: int = 1734
    num_filters: int = 256
    num_blocks: int = 19

    @nn.compact
    def __call__(self, board, marbles_out):
        # board shape: (batch, 9, 9)
        # marbles_out shape: (batch, 2)
        
        # Ajouter une dimension de canal au plateau
        x = board[..., None]  # (batch, 9, 9, 1)

        # Tronc commun
        x = nn.Conv(self.num_filters, (3, 3), padding='SAME')(x)
        x = nn.relu(x)
        
        for _ in range(self.num_blocks):
            x = ResBlock(self.num_filters)(x)

        # Aplatir les features spatiales
        x_flat = x.reshape((x.shape[0], -1))
        
        # Concaténer avec l'information des billes sorties
        marbles_out = jnp.broadcast_to(marbles_out, (x_flat.shape[0], 2))
        combined = jnp.concatenate([x_flat, marbles_out], axis=1)

        # Tête de politique (prior_logits)
        policy = jax.vmap(nn.Dense(1024))(combined)
        policy = nn.relu(policy)
        prior_logits = jax.vmap(nn.Dense(self.num_actions))(policy)
        # policy = nn.Dense(1024)(combined)
        # policy = nn.relu(policy)
        # prior_logits = nn.Dense(self.num_actions)(policy)
        
        # Tête de valeur
        value = nn.Dense(256)(combined)
        value = nn.relu(value)
        value = nn.Dense(1)(value)
        value = nn.tanh(value)  # Entre -1 et 1
        value = value.squeeze(-1)  # Enlever la dimension 1 pour avoir shape [batch_size]

        return prior_logits, value


class DummyAbaloneModel(nn.Module):
    """Réseau factice compatible avec l'architecture d'AbaloneModel"""
    num_actions: int = 1734

    @nn.compact
    def __call__(self, board, marbles_out):
        # Simule des prior logits uniformes et une valeur constante
        batch_size = board.shape[0]
        prior_logits = jnp.zeros((batch_size, self.num_actions))  # Priorités uniformes
        value = jnp.zeros((batch_size,))  # Valeur constante à 0
        return prior_logits, value


@partial(jax.jit, static_argnames=['radius'])
def prepare_input(board_3d: jnp.ndarray, our_marbles_out: int, opponent_marbles_out: int, radius: int = 4):
    # print("\nDébug prepare_input:")
    # print(f"Shape board_3d initial: {board_3d.shape}")
    
    # Convertir le plateau en 2D
    board_2d = cube_to_2d(board_3d, radius)
    # print(f"Shape après cube_to_2d: {board_2d.shape}")
    
    # Remplacer les NaN par 0
    board_2d = jnp.nan_to_num(board_2d, 0.0)
    
    # Ajouter la dimension de batch
    board_2d = board_2d[None, ...]
    # print(f"Shape final board_2d: {board_2d.shape}")
    
    # Créer le vecteur des billes sorties
    marbles_out = jnp.array([[our_marbles_out, opponent_marbles_out]])
    # print(f"Shape marbles_out: {marbles_out.shape}")
    
    return board_2d, marbles_out
def test_network():
    """
    Test du réseau avec une entrée aléatoire
    """
    # Créer une instance du modèle
    model = AbaloneModel()
    
    # Créer des données de test
    rng = jax.random.PRNGKey(0)
    batch_size = 1
    
    # Simuler un plateau et des compteurs de billes
    board = jax.random.uniform(rng, (batch_size, 9, 9))
    marbles_out = jax.random.uniform(rng, (batch_size, 2))
    
    # Initialiser les paramètres
    variables = model.init(rng, board, marbles_out)
    
    # Faire une prédiction
    policy, value = model.apply(variables, board, marbles_out)
    
    print("Test du réseau :")
    print(f"Forme du plateau en entrée : {board.shape}")
    print(f"Forme des compteurs en entrée : {marbles_out.shape}")
    print(f"Forme de la politique en sortie : {policy.shape}")
    print(f"Forme de la valeur en sortie : {value.shape}")
    
    # Vérifier que les dimensions sont correctes
    assert policy.shape == (batch_size, 1734), "La politique devrait être de taille (batch_size, 1734)"
   # assert value.shape == (batch_size, 1), "La valeur devrait être de taille (batch_size, 1)"
   # print("\nTest réussi ! Les dimensions sont correctes.")

if __name__ == "__main__":
    test_network()

# from env import AbaloneEnv
# def analyze_legal_moves():
#     """Analyse le nombre de coups légaux sur plusieurs positions"""
#     env = AbaloneEnv()
#     state = env.reset()
#     legal_moves = []
    
#     # Analyser quelques positions
#     for _ in range(1000):
#         moves = env.get_legal_moves(state)
#         num_legal = jnp.sum(moves)
#         legal_moves.append(num_legal)
        
#         # Faire un coup aléatoire pour la prochaine position
#         valid_moves = jnp.where(moves)[0]
#         action = valid_moves[0]  # Premier coup valide
#         state = env.step(state, action)
    
#     print("Statistiques des coups légaux :")
#     print(f"Min : {min(legal_moves)}")
#     print(f"Max : {max(legal_moves)}")
#     print(f"Moyenne : {sum(legal_moves)/len(legal_moves)}")

# if __name__ == "__main__":
#     analyze_legal_moves()
