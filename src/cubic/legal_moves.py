# legal_moves.py
import jax
import jax.numpy as jnp
import chex
from functools import partial
from typing import Dict, Tuple
from moves import move_single_marble, move_group_inline, move_group_parallel


@partial(jax.jit, static_argnames=['radius'])
def create_player_positions_mask(board: chex.Array, player: int, radius: int = 4) -> chex.Array:
    """
    Crée un masque booléen des positions où se trouvent les billes du joueur
    
    Args:
        board: État du plateau (2r+1, 2r+1, 2r+1)
        player: Joueur (1 ou -1)
        radius: Rayon du plateau
    
    Returns:
        mask: Tableau booléen de même taille que le plateau
    """
    return board == player

@partial(jax.jit, static_argnames=['radius'])
def filter_moves_by_positions(player_mask: chex.Array, 
                            moves_index: Dict[str, chex.Array],
                            radius: int = 4) -> chex.Array:
    """
    Crée un masque des mouvements dont toutes les positions de départ sont des billes du joueur
    """
    def check_move_positions(move_idx):
        debug_indices = [1425, 1428]
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
                   current_player: int,
                   radius: int = 4) -> chex.Array:
    """
    Détermine tous les mouvements légaux pour un état donné
    
    Args:
        board: État actuel du plateau
        moves_index: Index des mouvements possibles
        current_player: Joueur actuel (1 ou -1)
        radius: Rayon du plateau
        
    Returns:
        mask: Tableau booléen des mouvements légaux
    """
    # 1. Filtrer les mouvements selon la présence des pièces
    position_filtered = filter_moves_by_positions(
        create_player_positions_mask(board, current_player),
        moves_index,
        radius
    )
    
    # 2. Vérifier la légalité des mouvements filtrés
    return check_moves_validity(board, moves_index, position_filtered, radius)


############
def debug_filter_moves_by_positions(player_mask, moves_index, move_indices_to_debug, radius=4):
    """Version debug de filter_moves_by_positions qui ne vérifie que certains indices"""
    results = {}
    
    for move_idx in move_indices_to_debug:
        print(f"\nDébugging filter pour mouvement {move_idx}:")
        # Extraire les positions
        positions = moves_index['positions'][move_idx]
        group_size = moves_index['group_sizes'][move_idx]
        print(f"Positions: {positions}")
        print(f"Group size: {group_size}")
        
        # Convertir en indices du tableau
        board_positions = positions + radius
        print(f"Board positions: {board_positions}")
        
        # Vérifier la présence des billes aux positions
        has_pieces = player_mask[board_positions[:, 0],
                            board_positions[:, 1],
                            board_positions[:, 2]]
        print(f"Has pieces at positions: {has_pieces}")
        
        # Créer masque pour nombre correct de positions
        positions_mask = jnp.arange(3) < group_size
        print(f"Positions mask: {positions_mask}")
        
        # True si toutes les positions requises ont nos pièces
        result = jnp.all(jnp.where(positions_mask, has_pieces, True))
        print(f"Final filter result: {result}")
        
        results[move_idx] = result
    
    return results


def debug_check_moves_validity(board, moves_index, filtered_results, move_indices_to_debug, radius=4):
    """Version debug de check_moves_validity qui ne vérifie que certains indices"""
    results = {}
    
    for move_idx in move_indices_to_debug:
        print(f"\nDébugging validity pour mouvement {move_idx}:")
        
        # Si le mouvement n'a pas passé le premier filtre, on skippes
        is_filtered = filtered_results[move_idx]
        if not is_filtered:
            print("Mouvement non filtré, on passe")
            results[move_idx] = False
            continue
        
        # Récupérer les informations du mouvement
        positions = moves_index['positions'][move_idx]
        direction = moves_index['directions'][move_idx]
        move_type = moves_index['move_types'][move_idx]
        group_size = moves_index['group_sizes'][move_idx]
        
        print(f"Positions: {positions}")
        print(f"Direction: {direction}")
        print(f"Type de mouvement: {move_type} (0=single, 1=parallel, 2=inline)")
        print(f"Taille du groupe: {group_size}")
        
        # Vérifier les différents types de mouvements
        _, success_single = move_single_marble(board, positions[0], direction, radius)
        _, success_parallel = move_group_parallel(board, positions, direction, group_size, radius)
        _, success_inline, _ = move_group_inline(board, positions, direction, group_size, radius)
        
        print(f"Succès single: {success_single}")
        print(f"Succès parallel: {success_parallel}")
        print(f"Succès inline: {success_inline}")
        
        # Sélectionner le bon résultat selon le type
        is_valid = jnp.where(
            move_type == 0, success_single,
            jnp.where(move_type == 1, success_parallel, success_inline)
        )
        
        result = jnp.where(is_filtered, is_valid, False)
        print(f"Résultat final: {result}")
        
        results[move_idx] = result
    
    return results

def debug_specific_moves(board, moves_index, current_player, move_indices_to_debug=[1425, 1428], radius=4):
    """Version debug de get_legal_moves qui ne vérifie que certains indices"""
    print(f"Débugging des mouvements {move_indices_to_debug} pour le joueur {current_player}")
    
    # 1. Créer le masque des positions du joueur
    player_mask = create_player_positions_mask(board, current_player)
    
    # 2. Filtrer les mouvements selon la présence des pièces
    filtered_results = debug_filter_moves_by_positions(
        player_mask, moves_index, move_indices_to_debug, radius
    )
    
    # 3. Vérifier la légalité des mouvements filtrés
    final_results = debug_check_moves_validity(
        board, moves_index, filtered_results, move_indices_to_debug, radius
    )
    
    return final_results

