import jax
import jax.numpy as jnp
import chex
from typing import Tuple
import jax
import jax.numpy as jnp
import chex
from typing import Tuple
from functools import partial
from board import initialize_board, display_board


###########

@partial(jax.jit, static_argnames=['radius'])
def get_valid_positions(radius: int = 4):
    """
    Retourne une liste des positions valides dans le plateau hexagonal
    """
    return [
        # Ligne 0 (z = -4)
        (0,4,-4), (1,3,-4), (2,2,-4), (3,1,-4), (4,0,-4),
        # Ligne 1 (z = -3)
        (-1,4,-3), (0,3,-3), (1,2,-3), (2,1,-3), (3,0,-3), (4,-1,-3),
        # Ligne 2
        (-2,4,-2), (-1,3,-2), (0,2,-2), (1,1,-2), (2,0,-2), (3,-1,-2), (4,-2,-2),
        # Ligne 3
        (-3,4,-1), (-2,3,-1), (-1,2,-1), (0,1,-1), (1,0,-1), (2,-1,-1), (3,-2,-1), (4,-3,-1),
        #Ligne 4
        (-4,4,0), (-3,3,0), (-2,2,0), (-1,1,0), (0,0,0), (1,-1,0), (2,-2,0), (3,-3,0), (4,-4,0),
        # Ligne 5
        (-4,3,1), (-3,2,1), (-2,1,1), (-1,0,1), (0,-1,1), (1,-2,1), (2,-3,1), (3,-4,1),
        # Ligne 6
        (-4,2,2), (-3,1,2), (-2,0,2), (-1,-1,2), (0,-2,2), (1,-3,2), (2,-4,2),
        # Ligne 7
        (-4,1,3), (-3,0,3), (-2,-1,3), (-1,-2,3), (0,-3,3), (1,-4,3),
        # Ligne 8
        (-4,0,4), (-3,-1,4), (-2,-2,4), (-1,-3,4), (0,-4,4)
        # etc...
        # (ajouter toutes les positions valides)
    ]
@partial(jax.jit, static_argnames=['radius'])
def cube_to_2d(board_3d: chex.Array, radius: int = 4) -> chex.Array:
    """
    Convertit le plateau de la représentation cubique (3D) vers une grille 2D 9x9
    """
    board_2d = jnp.full((9, 9), jnp.nan)
    
    def convert_position(carry, position):
        board, = carry
        x, y, z = position
        
        # Convertir en indices de tableau 3D
        array_x = x + radius
        array_y = y + radius
        array_z = z + radius
        
        # Obtenir la valeur
        value = board_3d[array_x, array_y, array_z]
        
        # Calculer les coordonnées 2D
        row = z + radius
        col = x + 4  # Plus simple : on commence à la colonne 4 et on décale
        
        # Mettre à jour le tableau
        new_board = board.at[row, col].set(value)
        return (new_board,), None
    
    # Convertir toutes les positions valides
    (final_board,), _ = jax.lax.scan(convert_position, (board_2d,), jnp.array(get_valid_positions()))
    
    return final_board

def debug_cube_to_2d(board_3d: chex.Array, radius: int = 4) -> chex.Array:
    """
    Version de debug de cube_to_2d qui permet de voir les conversions
    """
    board_2d = jnp.full((9, 9), jnp.nan)
    positions = get_valid_positions()
    
    print("\nDébut conversion :")
    for x, y, z in positions:
        # Convertir en indices de tableau 3D
        array_x = x + radius
        array_y = y + radius
        array_z = z + radius
        
        # Obtenir la valeur
        value = board_3d[array_x, array_y, array_z]
        
        # Calculer les coordonnées 2D
        row = z + radius
        offset = 4 - row if row <= 4 else 0
        col = x + radius + offset
        
        print(f"Position cubique ({x},{y},{z}) -> Position tableau 3D ({array_x},{array_y},{array_z})")
        print(f"Valeur: {value}, Position 2D: (row={row},col={col})")
        
        board_2d = board_2d.at[row, col].set(value)
    
    return board_2d


def test_conversion_debug():
    """
    Test de la conversion avec debug
    """
    initial_board = initialize_board()
    print("\nPlateau initial 3D:")
    display_board(initial_board)
    
    board_2d = debug_cube_to_2d(initial_board)
    print("\nPlateau converti 2D:")
    display_2d_board(board_2d)

def display_2d_board(board_2d: chex.Array):
    """
    Affiche le plateau 2D
    """
    print("\nPlateau 2D:")
    for row in range(9):
        indent = abs(4 - row) if row <= 4 else 0
        print(" " * indent, end="")
        
        for col in range(9):
            value = board_2d[row, col]
            if jnp.isnan(value):
                print(" ", end=" ")
            elif value == 1:
                print("●", end=" ")
            elif value == -1:
                print("○", end=" ")
            else:
                print("·", end=" ")
        print()

####


def test_conversion():
    """
    Test de la conversion
    """
    initial_board = initialize_board()
    board_2d = cube_to_2d(initial_board)
    display_2d_board(board_2d)
# Test de la conversion
def test_conversion_pipeline():
    """
    Test complet du pipeline de conversion
    """
    # Créer un plateau initial
    initial_board = initialize_board()
    
    # Convertir en 2D
    board_2d = cube_to_2d(initial_board)
    
    print("Position initiale convertie en 2D:")
    display_2d_board(board_2d)
    print(board_2d)
    
    # Optionnel: afficher aussi la version originale pour comparaison
    print("\nPosition originale (pour comparaison):")
    display_board(initial_board)
import numpy as np
def check_moves_index():
    moves_data = np.load('move_map.npz')
    print(f"Nombre total de mouvements possibles : {len(moves_data['positions'])}")
    print("\nStructure du fichier:")
    for key in moves_data.files:
        print(f"{key}: shape = {moves_data[key].shape}")
    return len(moves_data['positions'])

if __name__ == '__main__':
   # test_conversion_pipeline()
    check_moves_index()
    #test_conversion_debug()