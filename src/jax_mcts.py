# src/jax_mcts.py
import chex
import jax.numpy as jnp
from mcts_utils import get_neighbors, get_all_valid_groups_jax, test_move_jax

@chex.dataclass(frozen=True)
class ActionMap:
    # Un tableau de shape (num_actions, 3) où chaque ligne contient [x, y, direction_index]
    actions: chex.Array
    # Un tableau de shape (num_actions,) contenant les indices
    indices: chex.Array
    num_actions: int


def create_action_map() -> ActionMap:
    """
    Crée le mapping de toutes les actions légales possibles dans Abalone.
    Returns:
        ActionMap: Le mapping complet des actions avec leurs indices
    """
    # Créer un plateau initial
    board = initialiser_plateau()
    all_actions = set()
    
    # Pour chaque joueur (noir et blanc)
    for player in [1, -1]:
        # Récupérer tous les groupes valides pour ce joueur
        valid_groups = get_all_valid_groups_jax(board, player)
        
        for group in valid_groups:
            if len(group) == 1:
                # Pour une seule bille, vérifier seulement les directions des voisins
                x, y = group[0]
                neighbors = get_neighbors(board, x, y)
                for direction in neighbors.keys():
                    if test_move_jax(board, [group[0]], direction, player):
                        all_actions.add((group, direction))
            else:
                # Pour les groupes de 2-3 billes, tester toutes les directions
                for direction in ['NW', 'NE', 'E', 'SE', 'SW', 'W']:
                    if test_move_jax(board, list(group), direction, player):
                        all_actions.add((group, direction))

    # Convertir en format pour MCTX
    direction_to_idx = {'NW': 0, 'NE': 1, 'E': 2, 'SE': 3, 'SW': 4, 'W': 5}
    actions_list = list(all_actions)
    
    return ActionMap(
        actions=jnp.array(actions_list),
        indices=jnp.arange(len(actions_list)),
        num_actions=len(actions_list)
    )

@chex.dataclass(frozen=True)
class AbaloneState:
    board: chex.Array  # Plateau 9x9
    current_player: chex.Array  # 1 ou -1
    legal_actions_mask: chex.Array  # Masque binaire des actions légales (1 si légal, 0 sinon)
    white_marbles_out: chex.Array
    black_marbles_out: chex.Array

def initialiser_plateau() -> chex.Array:
   """
   Crée et initialise le plateau de jeu d'Abalone avec JAX
   Returns:
       chex.Array: Le plateau initialisé avec les billes
   """
   # Créer un plateau 9x9 rempli de nan
   board = jnp.full((9, 9), jnp.nan)
   
   # D'abord remplir les cases valides avec 0 (cases vides)
   cells_per_row = [5, 6, 7, 8, 9, 8, 7, 6, 5]
   for y, n_cells in enumerate(cells_per_row):
       for x in range(n_cells):
           board = board.at[y, x].set(0)
   
   # Placement des billes noires (1) en haut
   # Ligne 0
   board = board.at[0, :5].set(1)
   # Ligne 1
   board = board.at[1, :6].set(1)
   # Ligne 2 : 2 espaces, 3 noires, 2 espaces
   board = board.at[2, 2:5].set(1)
   
   # Placement des billes blanches (-1) en bas
   # Ligne 8
   board = board.at[8, :5].set(-1)
   # Ligne 7
   board = board.at[7, :6].set(-1)
   # Ligne 6 : 2 espaces, 3 blanches, 2 espaces
   board = board.at[6, 2:5].set(-1)
   
   return board

def create_initial_state(action_map: ActionMap) -> AbaloneState:
    """Crée l'état initial du jeu"""
    # Création du plateau initial
    board = initialiser_plateau()  # à implémenter
    # Masque initial des actions légales
    legal_actions = jnp.zeros(action_map.num_actions)
    
    return AbaloneState(
        board=board,
        current_player=jnp.array(1),  # noir commence
        legal_actions_mask=legal_actions,
        white_marbles_out=jnp.array(0),
        black_marbles_out=jnp.array(0)
    )



if __name__ == "__main__":
    action_map = create_action_map()
    print(f"Nombre total d'actions légales : {action_map.num_actions}")
    print("\nExemples d'actions :")
    for i in range(5):
        print(f"Action {i}: {action_map.actions[i]}")
