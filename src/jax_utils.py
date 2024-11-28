# src/jax_utils.py
import jax
import jax.numpy as jnp

def get_cell_content_jax(board, x, y):
    """
    Vérifie le contenu d'une case avec JAX
    Returns:
        float: 1 (noir), -1 (blanc), 0 (vide), nan (invalide)
    """
    if x < 0 or y < 0 or x > 8 or y > 8:
        return float('nan')
    
    cells_per_row = [5, 6, 7, 8, 9, 8, 7, 6, 5]
    if x >= cells_per_row[y]:
        return float('nan')
    
    return board[y, x]

def get_neighbors(board, x, y):
    """
    Trouve les voisins valides d'une case donnée
    """
    neighbors = {}
    
    if y < 4:  # Partie haute
        directions = {
            'NW': (x-1, y-1),
            'NE': (x, y-1),
            'E': (x+1, y),
            'W': (x-1, y),
            'SW': (x, y+1),
            'SE': (x+1, y+1)
        }
    elif y == 4:  # Ligne du milieu
        directions = {
            'NW': (x-1, y-1),
            'NE': (x, y-1),
            'E': (x+1, y),
            'W': (x-1, y),
            'SW': (x-1, y+1),
            'SE': (x, y+1)
        }
    else:  # Partie basse
        directions = {
            'NW': (x, y-1),    
            'NE': (x+1, y-1),  
            'E': (x+1, y),
            'W': (x-1, y),
            'SW': (x-1, y+1),  
            'SE': (x, y+1)     
        }
    
    # Vérifier les voisins valides
    for direction, (nx, ny) in directions.items():
        content = get_cell_content_jax(board, nx, ny)
        if not jnp.isnan(content):  # Si la case est valide
            neighbors[direction] = (nx, ny)
            
    return neighbors

def is_valid_group(board, coordinates):
    """
    Vérifie si un groupe de coordonnées forme un groupe valide
    """
    if len(coordinates) < 2 or len(coordinates) > 3:
        return False, "Le groupe doit contenir 2 ou 3 billes", None
    
    # Vérifier que toutes les positions contiennent des billes de même couleur
    first_x, first_y = coordinates[0]
    color = get_cell_content_jax(board, first_x, first_y)
    if color == 0 or jnp.isnan(color):
        return False, "La première position ne contient pas de bille", None
    
    for x, y in coordinates[1:]:
        if get_cell_content_jax(board, x, y) != color:
            return False, "Les billes ne sont pas de la même couleur", None
    
    # Vérifier que les billes sont adjacentes et alignées
    alignment_direction = None
    for i, (x, y) in enumerate(coordinates):
        neighbors = get_neighbors(board, x, y)
        has_neighbor = False
        
        for j, (other_x, other_y) in enumerate(coordinates):
            if i != j:
                for direction, (nx, ny) in neighbors.items():
                    if nx == other_x and ny == other_y:
                        if alignment_direction is None:
                            alignment_direction = direction
                        elif alignment_direction != direction:
                            if not (
                                (alignment_direction == 'NW' and direction == 'SE') or
                                (alignment_direction == 'SE' and direction == 'NW') or
                                (alignment_direction == 'E' and direction == 'W') or
                                (alignment_direction == 'W' and direction == 'E') or
                                (alignment_direction == 'NE' and direction == 'SW') or
                                (alignment_direction == 'SW' and direction == 'NE')
                            ):
                                return False, "Les billes ne sont pas alignées", None
                        has_neighbor = True
                        break
        
        if not has_neighbor:
            return False, "Les billes ne sont pas toutes adjacentes", None
    
    return True, "Groupe valide", alignment_direction