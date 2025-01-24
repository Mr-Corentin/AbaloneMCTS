import jax
import jax.numpy as jnp
import chex
from functools import partial
from typing import Tuple

from core import CubeCoord, Direction, DIRECTIONS

################

def create_board_mask(radius: int = 4) -> chex.Array:
    """
    Crée un masque 3D des positions valides sur le plateau
    Returns:
        chex.Array: Masque booléen (2*radius+1, 2*radius+1, 2*radius+1)
    """
    size = 2 * radius + 1
    mask = jnp.full((size, size, size), False)
    
    # Liste de toutes les coordonnées valides
    valid_coords = [
        # Ligne 1 (z = -4)
        (0,4,-4), (1,3,-4), (2,2,-4), (3,1,-4), (4,0,-4),
        # Ligne 2 (z = -3)
        (-1,4,-3), (0,3,-3), (1,2,-3), (2,1,-3), (3,0,-3), (4,-1,-3),
        # Ligne 3 (z = -2)
        (-2,4,-2), (-1,3,-2), (0,2,-2), (1,1,-2), (2,0,-2), (3,-1,-2), (4,-2,-2),
        # Ligne 4 (z = -1)
        (-3,4,-1), (-2,3,-1), (-1,2,-1), (0,1,-1), (1,0,-1), (2,-1,-1), (3,-2,-1), (4,-3,-1),
        # Ligne 5 (z = 0)
        (-4,4,0), (-3,3,0), (-2,2,0), (-1,1,0), (0,0,0), (1,-1,0), (2,-2,0), (3,-3,0), (4,-4,0),
        # Ligne 6 (z = 1)
        (-4,3,1), (-3,2,1), (-2,1,1), (-1,0,1), (0,-1,1), (1,-2,1), (2,-3,1), (3,-4,1),
        # Ligne 7 (z = 2)
        (-4,2,2), (-3,1,2), (-2,0,2), (-1,-1,2), (0,-2,2), (1,-3,2), (2,-4,2),
        # Ligne 8 (z = 3)
        (-4,1,3), (-3,0,3), (-2,-1,3), (-1,-2,3), (0,-3,3), (1,-4,3),
        # Ligne 9 (z = 4)
        (-4,0,4), (-3,-1,4), (-2,-2,4), (-1,-3,4), (0,-4,4)
    ]
    
    # Marquer toutes les positions valides dans le masque
    for x, y, z in valid_coords:
        array_x = x + radius
        array_y = y + radius
        array_z = z + radius
        mask = mask.at[array_x, array_y, array_z].set(True)
    
    return mask

def initialize_board(radius: int = 4) -> chex.Array:
    """
    Initialise le plateau avec les positions de départ
    Returns:
        chex.Array: Plateau avec positions initiales
    """
    size = 2 * radius + 1
    board = jnp.full((size, size, size), jnp.nan)
    
    # Créer le masque des positions valides et initialiser les cases valides à 0
    valid_mask = create_board_mask(radius)
    board = jnp.where(valid_mask, 0., board)
    
    # Position initiale des billes noires (haut du plateau)
    black_coords = [
        # Première rangée
        (0,4,-4), (1,3,-4), (2,2,-4), (3,1,-4), (4,0,-4),
        # Deuxième rangée
        (-1,4,-3), (0,3,-3), (1,2,-3), (2,1,-3), (3,0,-3), (4,-1,-3),
        # Troisième rangée
        (0,2,-2), (1,1,-2), (2,0,-2)
    ]
    
    # Position initiale des billes blanches (bas du plateau)
    white_coords = [
        # Première rangée
        (-4,0,4), (-3,-1,4), (-2,-2,4), (-1,-3,4), (0,-4,4),
        # Deuxième rangée
        (-4,1,3), (-3,0,3), (-2,-1,3), (-1,-2,3), (0,-3,3), (1,-4,3),
        # Troisième rangée
        (-2,0,2), (-1,-1,2), (0,-2,2)
    ]
    
    # Placer les billes noires (1) et blanches (-1)
    for x, y, z in black_coords:
        array_x, array_y, array_z = x + radius, y + radius, z + radius
        board = board.at[array_x, array_y, array_z].set(1.)
    
    for x, y, z in white_coords:
        array_x, array_y, array_z = x + radius, y + radius, z + radius
        board = board.at[array_x, array_y, array_z].set(-1.)
    
    return board
def display_board(board: chex.Array, radius: int = 4):
    """
    Affiche le plateau Abalone en 2D en suivant les coordonnées cubiques
    """
    print("\nPlateau Abalone:")
    print("Légende: ● (noir), ○ (blanc), · (vide)\n")
    
    # Coordonnées de départ pour chaque ligne
    line_starts = [
        [(0,4,-4), 5],    # ligne 1: 5 cellules
        [(-1,4,-3), 6],   # ligne 2: 6 cellules
        [(-2,4,-2), 7],   # ligne 3: 7 cellules
        [(-3,4,-1), 8],   # ligne 4: 8 cellules
        [(-4,4,0), 9],    # ligne 5: 9 cellules
        [(-4,3,1), 8],    # ligne 6: 8 cellules
        [(-4,2,2), 7],    # ligne 7: 7 cellules
        [(-4,1,3), 6],    # ligne 8: 6 cellules
        [(-4,0,4), 5],    # ligne 9: 5 cellules
    ]
    
    # Pour chaque ligne
    for row_idx, ((start_x, start_y, start_z), num_cells) in enumerate(line_starts):
        # Indentation
        indent = abs(4 - row_idx)
        print(" " * indent, end="")
        
        # Parcourir les cellules de la ligne
        for i in range(num_cells):
            x = start_x + i
            y = start_y - i
            z = start_z
            
            array_x = x + radius
            array_y = y + radius
            array_z = z + radius
            
            value = board[array_x, array_y, array_z]
            if value == 1:
                print("● ", end="")
            elif value == -1:
                print("○ ", end="")
            else:
                print("· ", end="")
        print() 
                

def create_custom_board(marbles: list[tuple[tuple[int, int, int], int]], radius: int = 4) -> chex.Array:
    """
    Crée un plateau avec des billes placées à des positions spécifiques
    
    Args:
        marbles: Liste de tuples ((x,y,z), couleur) où couleur est 1 pour noir, -1 pour blanc
        radius: Rayon du plateau
    
    Returns:
        chex.Array: Plateau personnalisé
    """
    # Créer un plateau vide
    size = 2 * radius + 1
    board = jnp.full((size, size, size), jnp.nan)
    
    # Remplir toutes les positions valides avec 0 (vide)
    valid_mask = create_board_mask(radius)
    board = jnp.where(valid_mask, 0., board)
    
    # Placer les billes aux positions spécifiées
    for (x, y, z), color in marbles:
        board_pos = jnp.array([x, y, z]) + radius
        board = board.at[board_pos[0], board_pos[1], board_pos[2]].set(float(color))
    
    return board


# def verify_board(board: chex.Array, radius: int = 4):
#     """
#     Vérifie que le plateau est correctement initialisé
#     """
#     print("\nVérification du plateau :")
    
#     # Compter les billes
#     billes_noires = 0
#     billes_blanches = 0
#     cases_vides = 0
#     cases_invalides = 0
    
#     for x in range(-radius, radius+1):
#         for y in range(-radius, radius+1):
#             for z in range(-radius, radius+1):
#                 if x + y + z != 0:
#                     continue
                    
#                 array_x, array_y, array_z = x + radius, y + radius, z + radius
#                 valeur = board[array_x, array_y, array_z]
                
#                 if jnp.isnan(valeur):
#                     cases_invalides += 1
#                 elif valeur == 1:
#                     billes_noires += 1
#                 elif valeur == -1:
#                     billes_blanches += 1
#                 else:
#                     cases_vides += 1
    
#     print(f"Billes noires  : {billes_noires} (attendu: 14)")
#     print(f"Billes blanches: {billes_blanches} (attendu: 14)")
#     print(f"Cases vides    : {cases_vides} (attendu: 33)")
#     print(f"Total cases valides: {billes_noires + billes_blanches + cases_vides} (attendu: 61)\n")
    
#     # Vérifier quelques positions spécifiques
#     positions_test = [
#         ((0,4,-4), 1, "bille noire en haut"),
#         ((4,0,-4), 1, "bille noire en haut à droite"),
#         ((-4,0,4), -1, "bille blanche en bas"),
#         ((0,-4,4), -1, "bille blanche en bas à droite"),
#         ((0,0,0), 0, "centre du plateau")
#     ]
    
#     print("Vérification des positions clés:")
#     for (x, y, z), expected, desc in positions_test:
#         array_x, array_y, array_z = x + radius, y + radius, z + radius
#         valeur = board[array_x, array_y, array_z]
#         statut = "OK" if valeur == expected else "ERREUR"
#         print(f"Position ({x:2d},{y:2d},{z:2d}): {valeur:2.0f} (attendu: {expected:2d}) - {statut} - {desc}")

@jax.jit
def get_neighbors_array(pos_array: chex.Array) -> chex.Array:
    """
    Retourne les positions voisines d'une coordonnée donnée en format tableau
    Args:
        pos_array: Position d'origine (array [x, y, z])
    Returns:
        chex.Array: Tableau (6, 3) des coordonnées voisines
    """
    return pos_array[None, :] + DIRECTIONS

def get_neighbors(pos: CubeCoord) -> chex.Array:
    """
    Version wrapper qui accepte un CubeCoord
    """
    return get_neighbors_array(pos.to_array())

@partial(jax.jit, static_argnums=(2,))
def is_valid_position(pos: chex.Array, board: chex.Array, radius: int = 4) -> chex.Array:
    """
    Vérifie si une position est valide sur le plateau
    Args:
        pos: Position à vérifier (x, y, z)
        board: État du plateau
        radius: Rayon du plateau
    Returns:
        chex.Array: True si la position est valide
    """
    x, y, z = pos
    size = 2 * radius + 1
    
    # Utiliser jnp.where au lieu des opérations booléennes Python
    within_bounds = ((x + radius >= 0) & 
                    (x + radius < size) & 
                    (y + radius >= 0) & 
                    (y + radius < size) & 
                    (z + radius >= 0) & 
                    (z + radius < size))
    
    # Maintenant utiliser jnp.where pour combiner les conditions
    return jnp.where(
        within_bounds,
        ~jnp.isnan(board[x + radius, y + radius, z + radius]),
        False
    )


# def print_board_layer(board: chex.Array, layer: int):
#     """
#     Affiche une couche du plateau pour debug
#     Args:
#         board: Le plateau 3D
#         layer: L'indice de la couche z à afficher
#     """
#     radius = board.shape[0] // 2
#     print(f"Couche z={layer-radius}:")
#     for y in range(board.shape[1]):
#         for x in range(board.shape[0]):
#             val = board[x, y, layer]
#             if jnp.isnan(val):
#                 print(" . ", end="")
#             elif val == 1:
#                 print(" B ", end="")  # Bille noire
#             elif val == -1:
#                 print(" W ", end="")  # Bille blanche
#             else:
#                 print(" O ", end="")  # Case vide
#         print()

# def debug_board(board: chex.Array):
#     """Affiche toutes les couches du plateau"""
#     for z in range(board.shape[2]):
#         print_board_layer(board, z)
#         print()


#         print()  # Nouvelle ligne

# def verify_marble_positions(board: chex.Array, radius: int = 4):
#     """
#     Affiche les coordonnées cubiques de toutes les billes présentes sur le plateau
#     Args:
#         board: Le plateau 3D (2*radius+1, 2*radius+1, 2*radius+1)
#     """
#     print("\nPositions des billes :")
#     print("\nBilles noires (●) :")
#     black_found = False
#     white_found = False
    
#     # Parcourir tout le plateau
#     for x in range(-radius, radius+1):
#         for y in range(-radius, radius+1):
#             for z in range(-radius, radius+1):
#                 # Vérifier la contrainte x + y + z = 0
#                 if x + y + z != 0:
#                     continue
                    
#                 # Convertir en indices de tableau
#                 array_x = x + radius
#                 array_y = y + radius
#                 array_z = z + radius
                
#                 # Vérifier la valeur
#                 if 0 <= array_x < board.shape[0] and 0 <= array_y < board.shape[1] and 0 <= array_z < board.shape[2]:
#                     value = board[array_x, array_y, array_z]
#                     if value == 1:  # Bille noire
#                         print(f"({x:2d}, {y:2d}, {z:2d})")
#                         black_found = True
                        
#     print("\nBilles blanches (○) :")
#     for x in range(-radius, radius+1):
#         for y in range(-radius, radius+1):
#             for z in range(-radius, radius+1):
#                 if x + y + z != 0:
#                     continue
                    
#                 array_x = x + radius
#                 array_y = y + radius
#                 array_z = z + radius
                
#                 if 0 <= array_x < board.shape[0] and 0 <= array_y < board.shape[1] and 0 <= array_z < board.shape[2]:
#                     value = board[array_x, array_y, array_z]
#                     if value == -1:  # Bille blanche
#                         print(f"({x:2d}, {y:2d}, {z:2d})")
#                         white_found = True
    
#     if not black_found:
#         print("Aucune bille noire trouvée!")
#     if not white_found:
#         print("Aucune bille blanche trouvée!")



# def display_all_coordinates(board: chex.Array, radius: int = 4):
#     """
#     Affiche toutes les coordonnées valides du plateau dans l'ordre
#     """
#     cells_per_row = [5, 6, 7, 8, 9, 8, 7, 6, 5]  # De haut en bas
#     print("\nToutes les coordonnées valides du plateau:")
#     print("Légende: ● (noir/1), ○ (blanc/-1), · (vide/0)\n")

#     # Pour chaque ligne
#     for row_idx, num_cells in enumerate(cells_per_row):
#         z = row_idx - 4  # Conversion en coordonnée z
#         print(f"\nLigne {row_idx+1} (z = {z}), {num_cells} cellules:")
        
#         # Pour chaque colonne de cette ligne
#         start_x = -(num_cells // 2)
#         for i in range(num_cells):
#             x = start_x + i
#             y = -x - z  # Satisfaire x + y + z = 0
            
#             # Convertir en indices de tableau
#             array_x = x + radius
#             array_y = y + radius
#             array_z = z + radius
            
#             # Obtenir la valeur
#             value = board[array_x, array_y, array_z]
#             content = "●" if value == 1 else "○" if value == -1 else "·"
            
#             print(f"({x:2d}, {y:2d}, {z:2d}): {content}")
            
#     print("\nVérification des billes initiales:")
#     print("\nBilles noires attendues:")
#     expected_black = [
#         (0,4,-4), (1,3,-4), (2,2,-4), (3,1,-4), (4,0,-4),
#         (-1,4,-3), (0,3,-3), (1,2,-3), (2,1,-3), (3,0,-3), (4,-1,-3),
#         (0,2,-2), (1,1,-2), (2,0,-2)
#     ]
#     for x, y, z in expected_black:
#         array_x, array_y, array_z = x + radius, y + radius, z + radius
#         value = board[array_x, array_y, array_z]
#         print(f"({x:2d}, {y:2d}, {z:2d}): {'●' if value == 1 else 'x'}")

#     print("\nBilles blanches attendues:")
#     expected_white = [
#         (-4,0,4), (-3,-1,4), (-2,-2,4), (-1,-3,4), (0,-4,4),
#         (-4,1,3), (-3,0,3), (-2,-1,3), (-1,-2,3), (0,-3,3), (1,-4,3),
#         (-2,0,2), (-1,-1,2), (0,-2,2)
#     ]
#     for x, y, z in expected_white:
#         array_x, array_y, array_z = x + radius, y + radius, z + radius
#         value = board[array_x, array_y, array_z]
#         print(f"({x:2d}, {y:2d}, {z:2d}): {'○' if value == -1 else 'x'}")

