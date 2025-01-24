import jax
import jax.numpy as jnp
import chex
from functools import partial
from typing import Dict, Tuple
from moves_cano import move_single_marble, move_group_inline, move_group_parallel


@partial(jax.jit, static_argnames=['radius'])
def create_player_positions_mask(board: chex.Array, radius: int = 4) -> chex.Array:
    """
    Crée un masque booléen des positions où se trouvent les billes du joueur courant (toujours 1)
    """
    return board == 1

@partial(jax.jit, static_argnames=['radius'])
def filter_moves_by_positions(player_mask: chex.Array, 
                            moves_index: Dict[str, chex.Array],
                            radius: int = 4) -> chex.Array:
    """
    Crée un masque des mouvements dont toutes les positions de départ sont des billes du joueur
    """
    def check_move_positions(move_idx):
        # Extraire les positions
        positions = moves_index['positions'][move_idx]
        group_size = moves_index['group_sizes'][move_idx]
        
        # Convertir en indices du tableau
        board_positions = positions + radius
        
        # Vérifier la présence des billes aux positions
        has_pieces = player_mask[board_positions[:, 0],
                               board_positions[:, 1],
                               board_positions[:, 2]]
        
        # Créer masque pour nombre correct de positions
        positions_mask = jnp.arange(3) < group_size

       # True si toutes les positions requises ont nos pièces
        return jnp.all(jnp.where(positions_mask, has_pieces, True))
    
    return jax.vmap(check_move_positions)(jnp.arange(len(moves_index['directions'])))


@partial(jax.jit, static_argnames=['radius'])
def check_moves_validity(board: chex.Array,
                        moves_index: Dict[str, chex.Array],
                        filtered_moves: chex.Array,
                        radius: int = 4) -> chex.Array:
    """
    Vérifie quels mouvements filtrés sont légaux selon les règles du jeu
    """
    def check_move(move_idx):

        # Si le mouvement n'a pas passé le premier filtre, retourner False
        is_filtered = filtered_moves[move_idx]
        
        # Récupérer les informations du mouvement
        positions = moves_index['positions'][move_idx]
        direction = moves_index['directions'][move_idx]
        move_type = moves_index['move_types'][move_idx]
        group_size = moves_index['group_sizes'][move_idx]  # Récupérer la taille du groupe
        
        # Vérifier les différents types de mouvements
        # Pour mouvement simple
        _, success_single = move_single_marble(board, positions[0], direction, radius)
        
        # Pour mouvement parallel
        _, success_parallel = move_group_parallel(board, positions, direction, group_size, radius)
        
        # Pour mouvement inline
        _, success_inline, _ = move_group_inline(board, positions, direction, group_size, radius)
        
        # Sélectionner le bon résultat selon le type
        is_valid = jnp.where(
            move_type == 0, success_single,
            jnp.where(move_type == 1, success_parallel, success_inline)
        )

        # N'effectuer la vérification que si le mouvement est filtré
        return jnp.where(is_filtered, is_valid, False)
    
    # Vectoriser la vérification sur tous les mouvements
    return jax.vmap(check_move)(jnp.arange(len(moves_index['directions'])))


@partial(jax.jit, static_argnames=['radius'])
def get_legal_moves(board: chex.Array,
                   moves_index: Dict[str, chex.Array],
                   radius: int = 4) -> chex.Array:
    """
    Détermine tous les mouvements légaux pour le joueur courant (toujours 1)
    """
    position_filtered = filter_moves_by_positions(
        create_player_positions_mask(board),  # plus besoin de passer le 1
        moves_index,
        radius
    )
    
    return check_moves_validity(board, moves_index, position_filtered, radius)