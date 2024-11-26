# moves.py
from .utils import get_neighbors, is_valid_group
from .board import get_cell_content

def move_piece(board, start_x, start_y, end_x, end_y):
    """
    Déplace une seule bille d'une position à une autre
    Args:
        board: Le plateau de jeu
        start_x, start_y: Coordonnées de départ
        end_x, end_y: Coordonnées d'arrivée
    Returns:
        tuple: (succès, message)
    """
    # Vérifier si la case de départ contient une bille
    start_content = get_cell_content(board, start_x, start_y)
    if start_content not in ['W', 'B']:
        return False, "Pas de bille à déplacer sur la case de départ"
    
    # Vérifier si la case d'arrivée est valide et vide
    end_content = get_cell_content(board, end_x, end_y)
    if end_content not in ['O']:
        return False, "La case d'arrivée n'est pas valide ou n'est pas vide"
    
    # Vérifier si la case d'arrivée est un voisin valide
    neighbors = get_neighbors(board, start_x, start_y)
    destination = None
    for direction, (nx, ny) in neighbors.items():
        if nx == end_x and ny == end_y:
            destination = (nx, ny)
            break
    
    if destination is None:
        return False, "La case d'arrivée n'est pas un voisin valide"
    
    # Effectuer le déplacement
    board[end_y][end_x] = board[start_y][start_x]
    board[start_y][start_x] = 0
    
    return True, "Déplacement effectué avec succès"

def move_group_inline(board, coordinates, direction):
   """
   Déplace un groupe de billes dans la direction de leur alignement
   Args:
       board: le plateau de jeu
       coordinates: liste de tuples (x,y) des positions des billes
       direction: direction du mouvement ('NW', 'NE', 'E', 'SE', 'SW', 'W')
   Returns:
       (bool, str): (succès, message)
   """
   # Vérifier si le groupe est valide
   is_valid, message, alignment = is_valid_group(board, coordinates)
   if not is_valid:
       return False, message
   
   # Vérifier la validité du mouvement selon l'alignement
   # Si les billes sont sur la même colonne (même x)
   if len(set(x for x, y in coordinates)) == 1:
       if direction not in ['NW', 'SE'] and direction not in ['NE', 'SW']:
           print("Direction invalide pour un groupe vertical : doit être NW/SE ou NE/SW")
           return False, "Direction invalide pour un groupe vertical : doit être NW/SE ou NE/SW"
   
   # Si les billes sont sur la même ligne (même y)
   elif len(set(y for x, y in coordinates)) == 1:
       if direction not in ['E', 'W']:
           print("Direction invalide pour un groupe aligné horizontalement")
           return False, "Direction invalide pour un groupe aligné horizontalement"
   
   # Si les billes sont alignées diagonalement
   else:
       if alignment == 'NE' and direction not in ['NE', 'SW']:
           print("Direction invalide pour un groupe aligné en NE/SW")
           return False, "Direction invalide pour un groupe aligné en NE/SW"
       elif alignment == 'SE' and direction not in ['SE', 'NW']:
           print("Direction invalide pour un groupe aligné en SE/NW")
           return False, "Direction invalide pour un groupe aligné en SE/NW"
   
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
       return False, "Mouvement impossible : destination hors plateau"
   
   dest_x, dest_y = neighbors[direction]
   if get_cell_content(board, dest_x, dest_y) != 'O':
       return False, "La case de destination n'est pas libre"
   
   # Effectuer le déplacement
   piece_color = board[lead_y][lead_x]
   
   # Vider les positions initiales
   for x, y in coordinates:
       board[y][x] = 0
   
   # Placer les billes dans leurs nouvelles positions
   board[dest_y][dest_x] = piece_color
   
   for i in range(len(sorted_coordinates)-1):
       next_x, next_y = sorted_coordinates[i]
       board[next_y][next_x] = piece_color

   return True, "Mouvement en ligne effectué avec succès"

def move_group_parallel(board, coordinates, direction):
   """
   Déplace un groupe de billes parallèlement à leur alignement
   """
   # Vérifier si le groupe est valide
   is_valid, message, alignment = is_valid_group(board, coordinates)
   if not is_valid:
       return False, message
       
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
       content = get_cell_content(board, dest_x, dest_y)
       if content not in ['O']:
           return False, "Une ou plusieurs cases de destination sont occupées ou invalides"
   
   # Effectuer le déplacement
   piece_color = board[coordinates[0][1]][coordinates[0][0]]
   
   # D'abord vider toutes les positions de départ
   for x, y in coordinates:
       board[y][x] = 0
   
   # Puis remplir les destinations
   for dest_x, dest_y in destinations:
       board[dest_y][dest_x] = piece_color
   
   return True, "Mouvement parallèle effectué avec succès"

def is_valid_push(board, pushing_coordinates, direction):
    print(f"\nDébut vérification poussée:")
    print(f"Coordonnées initiales: {pushing_coordinates}")
    
    # Tri des coordonnées dans l'ordre de la poussée
    if direction == 'E':
        pushing_coordinates = sorted(pushing_coordinates, key=lambda pos: pos[0])
    elif direction == 'W':
        pushing_coordinates = sorted(pushing_coordinates, key=lambda pos: pos[0], reverse=True)
    elif direction in ['SE', 'SW']:
        pushing_coordinates = sorted(pushing_coordinates, key=lambda pos: pos[1])
    else:  # NW, NE
        pushing_coordinates = sorted(pushing_coordinates, key=lambda pos: pos[1], reverse=True)
    
    print(f"Coordonnées après tri: {pushing_coordinates}")
    
    # Vérification du groupe
    is_valid, message, alignment = is_valid_group(board, pushing_coordinates)
    print(f"Vérification groupe: valide={is_valid}, message={message}, alignment={alignment}")
    if not is_valid:
        return False, f"Groupe de billes qui poussent invalide: {message}", []
    
    # Récupérer la couleur des billes qui poussent
    pushing_color = get_cell_content(board, pushing_coordinates[0][0], pushing_coordinates[0][1])
    
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
    print(f"Position à pousser: ({next_x},{next_y})")
    
    # Identifier les billes qui vont être poussées
    pushed_coordinates = []
    current_x, current_y = next_x, next_y
    
    while True:
        content = get_cell_content(board, current_x, current_y)
        print(f"Vérification position ({current_x},{current_y}): contenu={content}")
        
        if content not in ['W', 'B']:  # Case vide ou invalide
            break
        if content == pushing_color:  # Bille amie
            return False, "Bille amie bloque la poussée", []
            
        pushed_coordinates.append((current_x, current_y))
        
        neighbors = get_neighbors(board, current_x, current_y)
        if direction not in neighbors:
            break
        current_x, current_y = neighbors[direction]
    
    print(f"Billes poussées trouvées: {pushed_coordinates}")
    
    # Vérifications finales
    if len(pushed_coordinates) == 0:
        return False, "Pas de billes à pousser", []
        
    if len(pushed_coordinates) >= len(pushing_coordinates):
        return False, "Pas assez de billes pour pousser", []
        
    return True, "Poussée valide", pushed_coordinates

def push_marbles(board, pushing_coordinates, direction):
    """
    Effectue une poussée de billes si elle est valide
    """
    # Vérifier d'abord si la poussée est valide
    is_valid, message, pushed_coordinates = is_valid_push(board, pushing_coordinates, direction)
    if not is_valid:
        return False, message, 0
    
    # Sauvegarder les couleurs
    pushing_color = board[pushing_coordinates[0][1]][pushing_coordinates[0][0]]
    pushed_color = board[pushed_coordinates[0][1]][pushed_coordinates[0][0]]
    
    # Calculer les nouvelles positions pour toutes les billes
    new_positions = []
    for x, y in pushing_coordinates:
        neighbors = get_neighbors(board, x, y)
        if direction not in neighbors:
            return False, "Mouvement impossible pour les billes qui poussent", 0
        new_positions.append(neighbors[direction])
    
    # Calculer le nombre de billes qui vont sortir
    billes_sorties = 0
    for x, y in pushed_coordinates:
        neighbors = get_neighbors(board, x, y)
        if direction not in neighbors:
            billes_sorties += 1
            
    # Vider toutes les positions initiales
    for x, y in pushing_coordinates + pushed_coordinates:
        board[y][x] = 0
        
    # Placer les billes poussées qui ne sortent pas du plateau
    for i, (x, y) in enumerate(pushed_coordinates):
        neighbors = get_neighbors(board, x, y)
        if direction in neighbors:
            next_x, next_y = neighbors[direction]
            board[next_y][next_x] = pushed_color
            
    # Placer les billes qui poussent dans leurs nouvelles positions
    for x, y in new_positions:
        board[y][x] = pushing_color
    
    return True, f"Poussée effectuée. {billes_sorties} billes sorties du plateau.", billes_sorties

def make_move(board, coordinates, direction):
    """
    Fonction générale pour effectuer un mouvement
    Args:
        board: le plateau de jeu
        coordinates: liste de tuples (x,y) des positions des billes à déplacer
        direction: direction du mouvement ('NW', 'NE', 'E', 'SE', 'SW', 'W')
    Returns:
        (bool, str): (succès, message)
    """
    # Vérifier si les coordonnées sont valides
    for x, y in coordinates:
        content = get_cell_content(board, x, y)
        if content not in ['W', 'B']:
            return False, f"Pas de bille en position ({x},{y})"
    
    # Cas d'une seule bille
    if len(coordinates) == 1:
        x, y = coordinates[0]
        return move_piece(board, x, y, *get_neighbors(board, x, y)[direction])
    
    # Cas d'un groupe (2 ou 3 billes)
    elif 2 <= len(coordinates) <= 3:
        is_valid, message, alignment = is_valid_group(board, coordinates)
        if not is_valid:
            return False, message
        
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
        elif alignment == 'NE' and direction in ['NE', 'SW']:  # Changé
            is_inline = True
        elif alignment == 'SE' and direction in ['SE', 'NW']:  # Changé
            is_inline = True
        if is_inline:
            return move_group_inline(board, coordinates, direction)
        else:
            return move_group_parallel(board, coordinates, direction)
    
    else:
        return False, "Nombre invalide de billes sélectionnées"