import jax
import jax.numpy as jnp
from functools import partial
import chex

DIRECTIONS = ['NW', 'NE', 'E', 'SE', 'SW', 'W']
DIR_TO_IDX = {dir: idx for idx, dir in enumerate(DIRECTIONS)}

MOVE_OFFSETS = {
    'upper': jnp.array([  # Pour y < 4
        [-1, -1],  # NW
        [0, -1],   # NE
        [1, 0],    # E
        [1, 1],    # SE
        [0, 1],    # SW
        [-1, 0],   # W
    ]),
    'middle': jnp.array([  # Pour y == 4
        [-1, -1],  # NW
        [0, -1],   # NE
        [1, 0],    # E
        [0, 1],    # SE
        [-1, 1],   # SW
        [-1, 0],   # W
    ]),
    'lower': jnp.array([  # Pour y > 4
        [0, -1],   # NW
        [1, -1],   # NE
        [1, 0],    # E
        [0, 1],    # SE
        [-1, 1],   # SW
        [-1, 0],   # W
    ])
}


def initialiser_plateau() -> chex.Array:
   """
   Crée et initialise le plateau de jeu d'Abalone avec JAX
   Returns:
       chex.Array: Le plateau initialisé avec les billes
   """
   board = jnp.full((9, 9), jnp.nan)
   
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


@partial(jax.jit, static_argnums=(1, 2))
def get_neighbors_jax(board: chex.Array, x: int, y: int) -> tuple[chex.Array, chex.Array]:
    """
    Trouve les voisins valides d'une case donnée de manière vectorisée
    Args:
        board: Le plateau de jeu (9x9)
        x, y: Coordonnées de la case
    Returns:
        tuple[chex.Array, chex.Array]: (masque des directions valides, coordonnées des voisins)
    """
    # Sélectionner le bon masque de déplacement
    offsets = jax.lax.select(
        y < 4,
        MOVE_OFFSETS['upper'],
        jax.lax.select(
            y == 4,
            MOVE_OFFSETS['middle'],
            MOVE_OFFSETS['lower']
        )
    )
    
    # Calculer toutes les positions voisines potentielles d'un coup
    neighbor_coords = jnp.array([x, y]) + offsets
    
    # Créer des masques pour les conditions de validité
    x_coords, y_coords = neighbor_coords[:, 0], neighbor_coords[:, 1]
    
    # Masque pour les coordonnées dans les limites du plateau
    in_bounds = (x_coords >= 0) & (x_coords < 9) & (y_coords >= 0) & (y_coords < 9)
    
    # Masque pour les cases valides (non-nan)
    cells_per_row = jnp.array([5, 6, 7, 8, 9, 8, 7, 6, 5])
    valid_x = x_coords < cells_per_row[y_coords]
    
    # Masque final combinant toutes les conditions
    valid_neighbors = in_bounds & valid_x
    
    return valid_neighbors, neighbor_coords
@partial(jax.jit, static_argnums=(1, 2))
def get_valid_neighbor_content(board: chex.Array, x: int, y: int) -> tuple[chex.Array, chex.Array, chex.Array]:
    """
    Retourne le contenu des cases voisines valides
    Args:
        board: Le plateau de jeu
        x, y: Coordonnées de la case
    Returns:
        tuple[chex.Array, chex.Array, chex.Array]: 
            - masque des voisins valides
            - coordonnées des voisins
            - contenu des cases voisines
    """
    valid_mask, neighbor_coords = get_neighbors_jax(board, x, y)
    
    # Extraire le contenu des cases voisines valides
    x_coords = neighbor_coords[:, 0].astype(jnp.int32)
    y_coords = neighbor_coords[:, 1].astype(jnp.int32)
    
    # Créer un masque pour l'indexation sécurisée
    safe_mask = valid_mask & (x_coords >= 0) & (x_coords < board.shape[1]) & \
                (y_coords >= 0) & (y_coords < board.shape[0])
    
    # Utiliser le masque pour une indexation sécurisée et obtenir un vecteur
    neighbor_content = jnp.where(
        safe_mask,
        board[y_coords, x_coords],
        jnp.nan
    )
    
    return valid_mask, neighbor_coords, neighbor_content

from functools import partial
import jax
import jax.numpy as jnp
import chex

# Constantes pour les directions
DIR_TO_IDX = {
    'NW': 0, 'NE': 1, 'E': 2, 
    'SE': 3, 'SW': 4, 'W': 5
}
IDX_TO_DIR = ['NW', 'NE', 'E', 'SE', 'SW', 'W']

@partial(jax.jit, static_argnames=['n_coords'])
def _is_valid_group_jax(board: chex.Array, coords: chex.Array, n_coords: int):
    """
    Version JIT de la validation de groupe
    """
    # Vérifier les couleurs
    colors = board[coords[:, 1], coords[:, 0]]
    same_color = jnp.all(colors == colors[0])
    valid_color = ~(jnp.isnan(colors[0])) & (colors[0] != 0)

    # Calculer les différences
    diff = coords[1:] - coords[:-1]
    
    # Différences valides pour l'adjacence
    valid_diffs = jnp.array([
        [-1, -1],  # NW (0)
        [0, -1],   # NE (1)
        [1, 0],    # E  (2)
        [1, 1],    # SE (3)
        [0, 1],    # SW (4)
        [-1, 0],   # W  (5)
    ])
    
    if n_coords == 2:
        # Pour 2 billes : vérifier l'adjacence
        matches = jnp.all(jnp.expand_dims(diff[0], 0) == valid_diffs, axis=1)
        is_adjacent = jnp.any(matches)
        direction_idx = jnp.argmax(matches)
        is_valid = is_adjacent & same_color & valid_color
        return is_valid, direction_idx
    else:
        # Pour 3 billes : vérifier l'alignement
        is_aligned = jnp.all(jnp.abs(diff[0]) == jnp.abs(diff[1]))
        dx, dy = diff[0]
        
        # Déterminer la direction comme dans l'ancienne implémentation
        direction_idx = jnp.where(dx == 0,
                                jnp.where(dy < 0, 0, 3),  # NW ou SE
                                jnp.where(dy == 0,
                                        jnp.where(dx > 0, 2, 5),  # E ou W
                                        3))  # Toujours SE pour les diagonales
        
        is_valid = is_aligned & same_color & valid_color
        return is_valid, direction_idx

def is_valid_group_wrapper(board: chex.Array, coordinates: list[tuple[int, int]]) -> tuple[bool, str, str]:
    """Wrapper pour garder l'interface originale"""
    if len(coordinates) < 2 or len(coordinates) > 3:
        return False, "Le groupe doit contenir 2 ou 3 billes", None
    
    coords = jnp.array(coordinates)
    try:
        is_valid, dir_idx = _is_valid_group_jax(board, coords, len(coordinates))
        # Convertir de JAX à Python
        is_valid = bool(is_valid)
        dir_idx = int(dir_idx)
        
        if is_valid:
            return True, "Groupe valide", IDX_TO_DIR[dir_idx]
        return False, "Groupe invalide", None
    except Exception as e:
        return False, f"Erreur: {str(e)}", None


@jax.jit
def get_cell_content_jax(board: chex.Array, x: int, y: int) -> chex.Array:
    """
    Vérifie le contenu d'une case avec JAX
    Args:
        board: Le plateau de jeu
        x, y: Coordonnées de la case
    Returns:
        chex.Array: 1 (noir), -1 (blanc), 0 (vide), nan (invalide)
    """
    # Conditions pour une position valide
    is_valid_x = (x >= 0) & (x < 9)
    is_valid_y = (y >= 0) & (y < 9)
    
    # Nombre de cellules valides par ligne
    cells_per_row = jnp.array([5, 6, 7, 8, 9, 8, 7, 6, 5])
    
    # Vérifier si la position est dans les limites de la ligne
    is_valid_cell = is_valid_x & is_valid_y & (x < cells_per_row[y])
    
    # Retourner le contenu si valide, sinon nan
    return jnp.where(
        is_valid_cell,
        board[y, x],
        jnp.nan
    )

@partial(jax.jit, static_argnums=(1,2,3,4))
def move_piece_jax(board: chex.Array, start_x: int, start_y: int, end_x: int, end_y: int) -> tuple[chex.Array, bool, int]:
    """
    Version JAX de move_piece qui déplace une seule bille
    Returns:
        tuple[chex.Array, bool, int]: (nouveau_plateau, succès, code_message)
        code_message: 0=succès, 1=case départ vide, 2=case arrivée invalide, 3=pas voisin
    """
    # Vérifier le contenu des cases
    start_content = board[start_y, start_x]
    valid_start = ~(jnp.isnan(start_content)) & (start_content != 0)
    
    # Vérifier la validité de la case d'arrivée
    end_content = jnp.where(
        (end_x >= 0) & (end_y >= 0) & (end_x < 9) & (end_y < 9),
        board[end_y, end_x],
        jnp.nan
    )
    valid_end = ~(jnp.isnan(end_content)) & (end_content == 0)
    
    # neighbors = jnp.array(get_neighbors(board, start_x, start_y))  # Temporaire, à remplacer
    # is_neighbor = jnp.any((neighbors[:, 0] == end_x) & (neighbors[:, 1] == end_y))
    
    # Vérifier que les cases sont voisines
    valid_mask, neighbor_coords = get_neighbors_jax(board, start_x, start_y)
    is_neighbor = jnp.any((neighbor_coords[:, 0] == end_x) & (neighbor_coords[:, 1] == end_y))
    # Conditions de succès
    success = valid_start & valid_end & is_neighbor
    
    # Code message
    msg_code = jnp.where(
        ~valid_start, 1,
        jnp.where(~valid_end, 2,
                 jnp.where(~is_neighbor, 3, 0))
    )
    
    # Créer le nouveau plateau
    new_board = board.at[end_y, end_x].set(
        jnp.where(success, start_content, board[end_y, end_x])
    )
    new_board = new_board.at[start_y, start_x].set(
        jnp.where(success, 0, board[start_y, start_x])
    )
    
    return new_board, success, msg_code

def move_piece_wrapper(board: chex.Array, start_x: int, start_y: int, end_x: int, end_y: int) -> tuple[chex.Array, bool, str]:
    """Wrapper pour garder l'interface originale"""
    try:
        new_board, success, msg_code = move_piece_jax(board, start_x, start_y, end_x, end_y)
        
        # Convertir le code en message
        messages = {
            0: "Déplacement effectué avec succès",
            1: "Pas de bille à déplacer sur la case de départ",
            2: "La case d'arrivée n'est pas valide ou n'est pas vide",
            3: "La case d'arrivée n'est pas un voisin valide"
        }
        
        return new_board, bool(success), messages[int(msg_code)]
    except Exception as e:
        return board, False, f"Erreur: {str(e)}"
    
@partial(jax.jit, static_argnames=['direction'])
def move_group_inline_jax(board: chex.Array, coords: chex.Array, direction: str) -> tuple[chex.Array, bool, int]:
    """
    Version JAX du déplacement en ligne d'un groupe de billes
    Args:
        board: Le plateau de jeu
        coords: Coordonnées des billes (Nx2)
        direction: Direction du mouvement
    Returns:
        tuple[chex.Array, bool, int]: (nouveau_plateau, succès, code_message)
    """
    # Vérifier si le groupe est valide
    is_valid, dir_idx = _is_valid_group_jax(board, coords, len(coords))
    
    # Vérifier la validité du mouvement selon l'alignement
    same_x = jnp.all(coords[:, 0] == coords[0, 0])
    same_y = jnp.all(coords[:, 1] == coords[0, 1])
    
    # Calculer les transformations selon la direction
    dir_transforms = {
        'E':  jnp.array([ 1,  0]),
        'W':  jnp.array([-1,  0]),
        'NE': jnp.array([ 1, -1]),
        'NW': jnp.array([-1, -1]),
        'SE': jnp.array([ 1,  1]),
        'SW': jnp.array([-1,  1]),
    }
    transform = dir_transforms[direction]
    
    # Valider la direction selon l'alignement
    valid_dir_vertical = same_x & ((direction in ['NW', 'SE', 'NE', 'SW']))
    valid_dir_horizontal = same_y & ((direction in ['E', 'W']))
    valid_dir_diagonal = (~same_x & ~same_y) & (
        ((dir_idx == DIR_TO_IDX['SE']) & (direction in ['SE', 'NW'])) |
        ((dir_idx == DIR_TO_IDX['NE']) & (direction in ['NE', 'SW']))
    )
    
    valid_direction = valid_dir_vertical | valid_dir_horizontal | valid_dir_diagonal
    
    # Trier les coordonnées selon la direction du mouvement
    if direction in ['E']:
        sorted_coords = coords[jnp.argsort(coords[:, 0])[::-1]]
    elif direction in ['W']:
        sorted_coords = coords[jnp.argsort(coords[:, 0])]
    elif direction in ['SE', 'SW']:
        sorted_coords = coords[jnp.argsort(coords[:, 1])[::-1]]
    else:  # NW, NE
        sorted_coords = coords[jnp.argsort(coords[:, 1])]
    
    # Vérifier la destination de la bille de tête
    lead_pos = sorted_coords[0]
    dest_pos = lead_pos + transform
    dest_content = get_cell_content_jax(board, dest_pos[0], dest_pos[1])
    valid_dest = (dest_content == 0)
    
    # Conditions finales de succès
    success = is_valid & valid_direction & valid_dest
    
    # Créer le nouveau plateau
    def move_pieces(board_state):
        # Sauvegarder la couleur des pièces
        piece_color = board[sorted_coords[0, 1], sorted_coords[0, 0]]
        
        # Vider les positions initiales
        for pos in sorted_coords:
            board_state = board_state.at[pos[1], pos[0]].set(0)
        
        # Placer les billes dans leurs nouvelles positions
        board_state = board_state.at[dest_pos[1], dest_pos[0]].set(piece_color)
        for i in range(len(sorted_coords)-1):
            pos = sorted_coords[i]
            board_state = board_state.at[pos[1], pos[0]].set(piece_color)
            
        return board_state
    
    new_board = jnp.where(success, move_pieces(board), board)
    
    # Code message
    msg_code = jnp.where(~is_valid, 1,
                        jnp.where(~valid_direction, 2,
                                jnp.where(~valid_dest, 3, 0)))
    
    return new_board, success, msg_code


def move_group_inline_wrapper(board: chex.Array, coordinates: list[tuple[int, int]], direction: str) -> tuple[chex.Array, bool, str]:
    """Wrapper pour garder l'interface originale"""
    try:
        coords = jnp.array(coordinates)
        new_board, success, msg_code = move_group_inline_jax(board, coords, direction)
        
        messages = {
            0: "Mouvement en ligne effectué avec succès",
            1: "Groupe de billes invalide",
            2: "Direction invalide pour ce type d'alignement",
            3: "Mouvement impossible ou destination invalide"
        }
        
        return new_board, bool(success), messages[int(msg_code)]
    except Exception as e:
        return board, False, f"Erreur: {str(e)}"
    


