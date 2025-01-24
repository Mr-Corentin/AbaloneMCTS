# src/jax_utils.py
import jax
import jax.numpy as jnp
from itertools import combinations
import chex


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


def move_piece(board, start_x, start_y, end_x, end_y):
    """
    Déplace une seule bille d'une position à une autre
    Args:
        board: Le plateau de jeu (jax array)
        start_x, start_y: Coordonnées de départ
        end_x, end_y: Coordonnées d'arrivée
    Returns:
        tuple: (nouveau_board, succès, message)
    """
    # Vérifier si la case de départ contient une bille
    start_content = get_cell_content_jax(board, start_x, start_y)
    if start_content == 0 or jnp.isnan(start_content):
        return board, False, "Pas de bille à déplacer sur la case de départ"
    
    # Vérifier si la case d'arrivée est valide et vide
    end_content = get_cell_content_jax(board, end_x, end_y)
    if end_content != 0:  # Si ce n'est pas une case vide
        return board, False, "La case d'arrivée n'est pas valide ou n'est pas vide"
    
    # Vérifier si la case d'arrivée est un voisin valide
    neighbors = get_neighbors(board, start_x, start_y)
    destination = None
    for direction, (nx, ny) in neighbors.items():
        if nx == end_x and ny == end_y:
            destination = (nx, ny)
            break
    
    if destination is None:
        return board, False, "La case d'arrivée n'est pas un voisin valide"
    
    # Effectuer le déplacement
    new_board = board.at[end_y, end_x].set(board[start_y, start_x])
    new_board = new_board.at[start_y, start_x].set(0)
    
    return new_board, True, "Déplacement effectué avec succès"

def move_group_inline(board, coordinates, direction):
    """
    Déplace un groupe de billes dans la direction de leur alignement
    Args:
        board: le plateau de jeu (jax array)
        coordinates: liste de tuples (x,y) des positions des billes
        direction: direction du mouvement ('NW', 'NE', 'E', 'SE', 'SW', 'W')
    Returns:
        tuple: (nouveau_board, succès, message)
    """
    # Vérifier si le groupe est valide
    is_valid, message, alignment = is_valid_group(board, coordinates)
    if not is_valid:
        return board, False, message
   
    # Vérifier la validité du mouvement selon l'alignement
    # Si les billes sont sur la même colonne (même x)
    if len(set(x for x, y in coordinates)) == 1:
        if direction not in ['NW', 'SE'] and direction not in ['NE', 'SW']:
            return board, False, "Direction invalide pour un groupe vertical : doit être NW/SE ou NE/SW"
   
    # Si les billes sont sur la même ligne (même y)
    elif len(set(y for x, y in coordinates)) == 1:
        if direction not in ['E', 'W']:
            return board, False, "Direction invalide pour un groupe aligné horizontalement"
   
    # Si les billes sont alignées diagonalement
    else:
        if alignment == 'NE' and direction not in ['NE', 'SW']:
            return board, False, "Direction invalide pour un groupe aligné en NE/SW"
        elif alignment == 'SE' and direction not in ['SE', 'NW']:
            return board, False, "Direction invalide pour un groupe aligné en SE/NW"
   
    # Trier les coordonnées selon la direction du mouvement
    if direction in ['E']:
        sorted_coordinates = sorted(coordinates, key=lambda pos: pos[0], reverse=True)
    elif direction in ['W']:
        sorted_coordinates = sorted(coordinates, key=lambda pos: pos[0])
    elif direction in ['SE', 'SW', 'NW', 'NE']:
        # Pour les mouvements diagonaux, on trie selon y
        if direction in ['SE', 'SW']:
            sorted_coordinates = sorted(coordinates, key=lambda pos: pos[1], reverse=True)
        else:  # NW, NE
            sorted_coordinates = sorted(coordinates, key=lambda pos: pos[1])
   
    # Vérifier si le mouvement est possible
    lead_x, lead_y = sorted_coordinates[0]
    neighbors = get_neighbors(board, lead_x, lead_y)
   
    if direction not in neighbors:
        return board, False, "Mouvement impossible : destination hors plateau"
   
    dest_x, dest_y = neighbors[direction]
    if get_cell_content_jax(board, dest_x, dest_y) != 0:
        return board, False, "La case de destination n'est pas libre"
   
    # Effectuer le déplacement
    piece_color = board[lead_y][lead_x]
    new_board = board
   
    # Vider les positions initiales
    for x, y in coordinates:
        new_board = new_board.at[y, x].set(0)
   
    # Placer les billes dans leurs nouvelles positions
    new_board = new_board.at[dest_y, dest_x].set(piece_color)
   
    for i in range(len(sorted_coordinates)-1):
        next_x, next_y = sorted_coordinates[i]
        new_board = new_board.at[next_y, next_x].set(piece_color)

    return new_board, True, "Mouvement en ligne effectué avec succès"

def move_group_parallel(board, coordinates, direction):
   """
   Déplace un groupe de billes parallèlement à leur alignement
   Args:
       board: le plateau de jeu (jax array)
       coordinates: liste de tuples (x,y) des positions des billes
       direction: direction du mouvement
   Returns:
       tuple: (nouveau_board, succès, message)
   """
   # Vérifier si le groupe est valide
   is_valid, message, alignment = is_valid_group(board, coordinates)
   if not is_valid:
       return board, False, message
       
   # Calculer les nouvelles coordonnées selon la direction
   destinations = []
   
   # Calculer les destinations pour chaque bille selon sa position sur le plateau
   for x, y in coordinates:
       if y < 4:  # Partie haute
           if direction == 'NW':
               destinations.append((x-1, y-1))
           elif direction == 'NE':
               destinations.append((x, y-1))
           elif direction == 'E':
               destinations.append((x+1, y))
           elif direction == 'W':
               destinations.append((x-1, y))
           elif direction == 'SW':
               destinations.append((x, y+1))
           elif direction == 'SE':
               destinations.append((x+1, y+1))
       elif y == 4:  # Ligne du milieu
           if direction == 'NW':
               destinations.append((x-1, y-1))
           elif direction == 'NE':
               destinations.append((x, y-1))
           elif direction == 'E':
               destinations.append((x+1, y))
           elif direction == 'W':
               destinations.append((x-1, y))
           elif direction == 'SW':
               destinations.append((x-1, y+1))
           elif direction == 'SE':
               destinations.append((x, y+1))
       else:  # Partie basse
           if direction == 'NW':
               destinations.append((x, y-1))
           elif direction == 'NE':
               destinations.append((x+1, y-1))
           elif direction == 'E':
               destinations.append((x+1, y))
           elif direction == 'W':
               destinations.append((x-1, y))
           elif direction == 'SW':
               destinations.append((x-1, y+1))
           elif direction == 'SE':
               destinations.append((x, y+1))
   
   # Vérifier que toutes les destinations sont valides
   for dest_x, dest_y in destinations:
       content = get_cell_content_jax(board, dest_x, dest_y)
       if content != 0:
           return board, False, "Une ou plusieurs cases de destination sont occupées ou invalides"
   
   # Effectuer le déplacement
   piece_color = board[coordinates[0][1]][coordinates[0][0]]
   new_board = board
   
   # D'abord vider toutes les positions de départ
   for x, y in coordinates:
       new_board = new_board.at[y, x].set(0)
   
   # Puis remplir les destinations
   for dest_x, dest_y in destinations:
       new_board = new_board.at[dest_y, dest_x].set(piece_color)
   
   return new_board, True, "Mouvement parallèle effectué avec succès"

def is_valid_push(board, pushing_coordinates, direction):
   """
   Vérifie si une poussée est valide et identifie les billes poussées
   Args:
       board: Le plateau de jeu (jax array)
       pushing_coordinates: Liste de tuples (x,y) des billes qui poussent
       direction: Direction de la poussée
   Returns:
       tuple: (est_valide, message, billes_poussées)
   """
   # Tri des coordonnées dans l'ordre de la poussée
   if direction == 'E':
       pushing_coordinates = sorted(pushing_coordinates, key=lambda pos: pos[0])
   elif direction == 'W':
       pushing_coordinates = sorted(pushing_coordinates, key=lambda pos: pos[0], reverse=True)
   elif direction in ['SE', 'SW']:
       pushing_coordinates = sorted(pushing_coordinates, key=lambda pos: pos[1])
   else:  # NW, NE
       pushing_coordinates = sorted(pushing_coordinates, key=lambda pos: pos[1], reverse=True)
   
   # Vérification du groupe
   is_valid, message, alignment = is_valid_group(board, pushing_coordinates)
   if not is_valid:
       return False, f"Groupe de billes qui poussent invalide: {message}", []
   
   # Récupérer la couleur des billes qui poussent
   pushing_color = get_cell_content_jax(board, pushing_coordinates[0][0], pushing_coordinates[0][1])
   
   # Vérifier que toutes les positions entre la première et la dernière bille sont occupées par nos billes
   for i in range(len(pushing_coordinates)-1):
       current = pushing_coordinates[i]
       next_pos = pushing_coordinates[i+1]
       neighbors = get_neighbors(board, current[0], current[1])
       if direction not in neighbors or neighbors[direction] != next_pos:
           return False, "Les billes qui poussent ne sont pas adjacentes dans la direction de poussée", []
   
   # Trouver la position après la dernière bille qui pousse
   last_x, last_y = pushing_coordinates[-1]
   neighbors = get_neighbors(board, last_x, last_y)
   if direction not in neighbors:
       return False, "Pas de case valide dans cette direction", []
   
   next_x, next_y = neighbors[direction]
   
   # Identifier les billes qui vont être poussées
   pushed_coordinates = []
   current_x, current_y = next_x, next_y
   
   while True:
       content = get_cell_content_jax(board, current_x, current_y)
       
       if content == 0 or jnp.isnan(content):  # Case vide ou invalide
           break
       if content == pushing_color:  # Bille amie
           return False, "Bille amie bloque la poussée", []
           
       pushed_coordinates.append((current_x, current_y))
       
       neighbors = get_neighbors(board, current_x, current_y)
       if direction not in neighbors:
           break
       current_x, current_y = neighbors[direction]
   
   # Vérifications finales
   if len(pushed_coordinates) == 0:
       return False, "Pas de billes à pousser", []
       
   if len(pushed_coordinates) >= len(pushing_coordinates):
       return False, "Pas assez de billes pour pousser", []
   
   return True, "Poussée valide", pushed_coordinates

def push_marbles(board, pushing_coordinates, direction):
   """
   Effectue une poussée de billes si elle est valide
   Args:
       board: Le plateau de jeu (jax array)
       pushing_coordinates: Liste de tuples (x,y) des billes qui poussent
       direction: Direction de la poussée
   Returns:
       tuple: (nouveau_board, succès, message, nombre_billes_sorties)
   """
   # Vérifier d'abord si la poussée est valide
   is_valid, message, pushed_coordinates = is_valid_push(board, pushing_coordinates, direction)
   if not is_valid:
       return board, False, message, 0
   
   # Sauvegarder les couleurs
   pushing_color = board[pushing_coordinates[0][1]][pushing_coordinates[0][0]]
   pushed_color = board[pushed_coordinates[0][1]][pushed_coordinates[0][0]]
   
   # Calculer les nouvelles positions pour toutes les billes
   new_positions = []
   for x, y in pushing_coordinates:
       neighbors = get_neighbors(board, x, y)
       if direction not in neighbors:
           return board, False, "Mouvement impossible pour les billes qui poussent", 0
       new_positions.append(neighbors[direction])
   
   # Calculer le nombre de billes qui vont sortir
   billes_sorties = 0
   for x, y in pushed_coordinates:
       neighbors = get_neighbors(board, x, y)
       if direction not in neighbors:
           billes_sorties += 1
           
   # Créer le nouveau plateau et effectuer les modifications
   new_board = board
   
   # Vider toutes les positions initiales
   for x, y in pushing_coordinates + pushed_coordinates:
       new_board = new_board.at[y, x].set(0)
       
   # Placer les billes poussées qui ne sortent pas du plateau
   for i, (x, y) in enumerate(pushed_coordinates):
       neighbors = get_neighbors(board, x, y)
       if direction in neighbors:
           next_x, next_y = neighbors[direction]
           new_board = new_board.at[next_y, next_x].set(pushed_color)
           
   # Placer les billes qui poussent dans leurs nouvelles positions
   for x, y in new_positions:
       new_board = new_board.at[y, x].set(pushing_color)
   
   return new_board, True, f"Poussée effectuée. {billes_sorties} billes sorties du plateau.", billes_sorties

def make_move(board, coordinates, direction):
   """
   Fonction générale pour effectuer un mouvement
   Args:
       board: le plateau de jeu (jax array)
       coordinates: liste de tuples (x,y) des positions des billes à déplacer
       direction: direction du mouvement ('NW', 'NE', 'E', 'SE', 'SW', 'W')
   Returns:
       tuple: (nouveau_board, succès, message)
   """
   # Vérifier si les coordonnées sont valides
   for x, y in coordinates:
       content = get_cell_content_jax(board, x, y)
       if content == 0 or jnp.isnan(content):
           return board, False, f"Pas de bille en position ({x},{y})"
   
   # Cas d'une seule bille
   if len(coordinates) == 1:
       x, y = coordinates[0]
       neighbors = get_neighbors(board, x, y)
       if direction not in neighbors:
           return board, False, "Direction invalide"
       end_x, end_y = neighbors[direction]
       return move_piece(board, x, y, end_x, end_y)
   
   # Cas d'un groupe (2 ou 3 billes)
   elif 2 <= len(coordinates) <= 3:
       is_valid, message, alignment = is_valid_group(board, coordinates)
       if not is_valid:
           return board, False, message
       
       # Standardiser l'alignement
       if alignment == 'W':
           alignment = 'E'
       elif alignment == 'SW':
           alignment = 'NE'
       elif alignment == 'NW':
           alignment = 'SE'
       
       # Vérifier si c'est un mouvement inline
       is_inline = False
       if alignment == 'E' and direction in ['E', 'W']:
           is_inline = True
       elif alignment == 'NE' and direction in ['NE', 'SW']:
           is_inline = True
       elif alignment == 'SE' and direction in ['SE', 'NW']:
           is_inline = True
           
       # Tester si c'est une poussée valide
       new_board, success, message, billes_sorties = push_marbles(board, coordinates, direction)
       if success:
           return new_board, True, message
       
       # Si ce n'est pas une poussée, essayer un mouvement normal
       if is_inline:
           return move_group_inline(board, coordinates, direction)
       else:
           return move_group_parallel(board, coordinates, direction)
   
   else:
       return board, False, "Nombre invalide de billes sélectionnées"
   

def get_all_valid_groups_jax(board, current_player):
    """
    Trouve tous les groupes valides de billes pour le joueur actuel
    Args:
        board: jax array du plateau
        current_player: 1 pour noir, -1 pour blanc
    Returns:
        list: Liste de tuples de coordonnées représentant les groupes valides
    """
    valid_groups = []
    
    # Trouver toutes les billes du joueur actuel
    player_marbles = []
    for y in range(9):
        for x in range(9):
            content = get_cell_content_jax(board, x, y)
            if content == current_player:  # 1 ou -1
                player_marbles.append((x, y))
    
    # D'abord ajouter les groupes de 1 bille
    valid_groups.extend([(marble,) for marble in player_marbles])

    # Tester les combinaisons de 2 et 3 billes
    for size in range(2, 4):
        for marble_group in combinations(player_marbles, size):
            is_valid, _, alignment = is_valid_group(board, marble_group)
            if is_valid:
                valid_groups.append(marble_group)
    
    return valid_groups

def test_move_jax(board, coordinates, direction, player):
    """
    Teste si un mouvement est valide sans modifier l'état du jeu
    Args:
        board: jax array du plateau
        coordinates: liste de tuples (x,y) des positions des billes
        direction: direction du mouvement
        player: joueur actuel (1 pour noir, -1 pour blanc)
    Returns:
        bool: True si le mouvement est valide
    """
    # Vérifier que les billes appartiennent au joueur
    for x, y in coordinates:
        if get_cell_content_jax(board, x, y) != player:
            return False

    # Essayer une poussée
    new_board, success, _, _ = push_marbles(board, coordinates, direction)
    if success:
        return True

    # Si ce n'est pas une poussée, essayer un déplacement normal
    if len(coordinates) <= 3:
        new_board, success, _ = make_move(board, coordinates, direction)
        if success:
            return True

    return False