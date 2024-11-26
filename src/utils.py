# utils.py
from .board import get_cell_content

def get_neighbors(board, x, y):
    """
    Trouve les voisins valides d'une case donnée
    Args:
        board: Le plateau de jeu
        x, y: Coordonnées de la case
    Returns:
        dict: Dictionnaire des voisins valides avec leurs directions
    """
    neighbors = {}
    
    # Partie haute du plateau (lignes 0-4)
    if y < 4:
        directions = {
            'NW': (x-1, y-1),
            'NE': (x, y-1),
            'E': (x+1, y),
            'W': (x-1, y),
            'SW': (x, y+1),
            'SE': (x+1, y+1)
        }
    elif y == 4:
        directions = {
            'NW': (x-1, y-1),
            'NE': (x, y-1),
            'E': (x+1, y),
            'W': (x-1, y),
            'SW': (x-1, y+1),
            'SE': (x, y+1)
        }
    else:  # Partie basse du plateau (lignes 5-8)
        directions = {
            'NW': (x, y-1),    
            'NE': (x+1, y-1),  
            'E': (x+1, y),
            'W': (x-1, y),
            'SW': (x-1, y+1),  
            'SE': (x, y+1)     
        }
    
    # On vérifie quels voisins sont valides
    for direction, (nx, ny) in directions.items():
        content = get_cell_content(board, nx, ny)
        if content in ['W', 'B', 'O']: 
            neighbors[direction] = (nx, ny)
            
    return neighbors

def is_valid_group(board, coordinates):
    """
    Vérifie si un groupe de coordonnées forme un groupe valide
    Args:
        board: Le plateau de jeu
        coordinates: Liste de tuples (x,y) représentant les positions des billes
    Returns:
        tuple: (est_valide, message, direction_alignement)
    """
    # Vérifier le nombre de billes
    if len(coordinates) < 2 or len(coordinates) > 3:
        return False, "Le groupe doit contenir 2 ou 3 billes", None
    
    # Vérifier que toutes les positions contiennent des billes de même couleur
    first_x, first_y = coordinates[0]
    color = get_cell_content(board, first_x, first_y)
    if color not in ['W', 'B']:
        return False, "La première position ne contient pas de bille", None
    
    for x, y in coordinates[1:]:
        if get_cell_content(board, x, y) != color:
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
                            # Vérifier si la direction est opposée
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