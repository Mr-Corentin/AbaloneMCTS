import jax
import jax.numpy as jnp
import chex
from functools import partial
from core import Direction, DIRECTIONS
from positions import get_valid_neighbors, analyze_group

# Dans moves.py
@partial(jax.jit, static_argnames=['radius', 'direction_idx'])
def move_single_marble(board: chex.Array, 
                      position: chex.Array, 
                      direction_idx: int,
                      radius: int = 4) -> tuple[chex.Array, bool]:
    """
    Déplace une seule bille du joueur courant (toujours 1) dans une direction donnée.
    
    Args:
        board: État actuel du plateau (joueur courant = 1)
        position: Position de départ [x, y, z]
        direction_idx: Index de la direction (0-5)
        radius: Rayon du plateau
    
    Returns:
        tuple[chex.Array, bool]: (nouveau plateau, succès du mouvement)
    """
    # Obtenir la valeur de la position de départ
    pos_idx = position + radius
    start_value = board[pos_idx[0], pos_idx[1], pos_idx[2]]
    
    # Vérifier que la position contient une bille du joueur courant
    has_marble = start_value == 1  # Simplifié car toujours 1
    
    # Obtenir le vecteur de direction
    dir_vec = DIRECTIONS[direction_idx]
    new_position = position + dir_vec
    new_pos_idx = new_position + radius
    
    # Vérifier si la nouvelle position est valide et vide
    valid_mask, _ = get_valid_neighbors(position, board, radius)
    is_valid_move = valid_mask[direction_idx]
    dest_value = board[new_pos_idx[0], new_pos_idx[1], new_pos_idx[2]]
    is_empty = (dest_value == 0)
    
    # Succès si tout est valide
    success = has_marble & is_valid_move & is_empty
    
    # Créer le nouveau plateau
    new_board = board.at[pos_idx[0], pos_idx[1], pos_idx[2]].set(
        jnp.where(success, 0, start_value)
    )
    new_board = new_board.at[new_pos_idx[0], new_pos_idx[1], new_pos_idx[2]].set(
        jnp.where(success, 1, dest_value)  # Toujours 1
    )
    
    return new_board, success
@partial(jax.jit, static_argnames=['radius'])
def move_group_parallel(board: chex.Array, 
                       positions: chex.Array,
                       direction: int,
                       group_size: int,
                       radius: int = 4) -> tuple[chex.Array, bool]:
    """
    Déplace un groupe de billes du joueur courant (toujours 1) parallèlement à leur alignement
    """
    # Créer le masque pour les positions valides
    positions_mask = jnp.arange(positions.shape[0]) < group_size
    valid_positions = jnp.where(positions_mask[:, None], positions, 0)
    
    # Analyser le groupe
    is_valid, inline_dirs, parallel_dirs = analyze_group(valid_positions, board, group_size, radius)
    
    # Vérifier que la direction est un mouvement parallèle valide
    is_valid_direction = parallel_dirs[direction]
    
    # Calculer les nouvelles positions
    dir_vec = DIRECTIONS[direction]
    new_positions = valid_positions + dir_vec
    board_positions = new_positions + radius
    
    # Vérifier que les positions sont dans les limites
    in_bounds = jnp.all((board_positions >= 0) & (board_positions < board.shape[0]))
    
    # Vérifier que les destinations sont vides
    destinations_empty = jnp.all(jnp.where(
        positions_mask,
        board[board_positions[:, 0], board_positions[:, 1], board_positions[:, 2]] == 0,
        True
    ))
    
    # Succès si tout est valide
    success = is_valid & is_valid_direction & in_bounds & destinations_empty
    
    # Plus besoin de vérifier piece_color, c'est toujours 1
    idx_mask = jnp.arange(3) < group_size
    
    def update_board(current_board):
        # Vider les anciennes positions
        old_positions = valid_positions + radius
        for_old_pos = current_board.at[
            old_positions[:, 0],
            old_positions[:, 1],
            old_positions[:, 2]
        ].set(jnp.where(
            idx_mask,
            jnp.where(success, 0, current_board[
                old_positions[:, 0],
                old_positions[:, 1],
                old_positions[:, 2]
            ]),
            current_board[
                old_positions[:, 0],
                old_positions[:, 1],
                old_positions[:, 2]
            ]
        ))
        
        # Remplir les nouvelles positions avec 1 (joueur courant)
        return for_old_pos.at[
            board_positions[:, 0],
            board_positions[:, 1],
            board_positions[:, 2]
        ].set(jnp.where(
            idx_mask,
            jnp.where(success, 1, current_board[  # Directement 1 au lieu de piece_color
                board_positions[:, 0],
                board_positions[:, 1],
                board_positions[:, 2]
            ]),
            for_old_pos[
                board_positions[:, 0],
                board_positions[:, 1],
                board_positions[:, 2]
            ]
        ))
    
    new_board = update_board(board)
    
    return new_board, success

@partial(jax.jit, static_argnames=['radius'])
def move_group_inline_old(board: chex.Array, 
                    positions: chex.Array,
                    direction: int,
                    group_size: int,
                    radius: int = 4) -> tuple[chex.Array, bool, int]:
    """
    Déplace un groupe de billes dans la direction de leur alignement
    Gère à la fois les déplacements simples et les poussées
    Le joueur courant est toujours représenté par 1, l'adversaire par -1
    """
    # 1. Créer le masque pour les positions valides
    positions_mask = jnp.arange(positions.shape[0]) < group_size
    valid_positions = jnp.where(positions_mask[:, None], positions, 0)
    
    # 2. Analyser le groupe
    is_valid, inline_dirs, _ = analyze_group(valid_positions, board, group_size, radius)
    is_valid_direction = inline_dirs[direction]

    # 3. Plus besoin de obtenir la couleur car c'est toujours 1
    start_positions = valid_positions + radius

    # 4. Calculer les nouvelles positions et la position devant le groupe
    dir_vec = DIRECTIONS[direction]
    new_positions = positions + dir_vec
   # #print("new_pos",new_positions)
    board_positions = new_positions + radius

    # 5. Vérifier si les nouvelles positions sont dans les limites
    in_bounds = jnp.all((board_positions >= 0) & (board_positions < board.shape[0]))

    # 6. Analyser ce qui se trouve devant le groupe
    front_pos = valid_positions[group_size-1]  # Dernière bille de notre groupe
    push_positions = front_pos + jnp.array([dir_vec * (i + 1) for i in range(3)])
    push_board_positions = push_positions + radius


    scores = jnp.where(
        direction == 1,  # Est (x+1, y-1, z)
        valid_positions[:, 0] + (-valid_positions[:, 1]),  # maximise x, minimise y
        jnp.where(
            direction == 4,  # Ouest (x-1, y+1, z)
            -valid_positions[:, 0] + valid_positions[:, 1],  # minimise x, maximise y
            jnp.where(
                direction == 0,  # Nord-Est (x+1, y, z-1)
                valid_positions[:, 0] + (-valid_positions[:, 2]),  # maximise x, minimise z
                jnp.where(
                    direction == 3,  # Sud-Ouest (x-1, y, z+1)
                    -valid_positions[:, 0] + valid_positions[:, 2],  # minimise x, maximise z
                    jnp.where(
                        direction == 2,  # Sud-Est (x, y-1, z+1)
                        -valid_positions[:, 1] + valid_positions[:, 2],  # minimise y, maximise z
                        valid_positions[:, 1] + (-valid_positions[:, 2])  # NO: maximise y, minimise z
                    )
                )
            )
        )
    )

    scores = jnp.where(positions_mask, scores, -jnp.inf)
    head_index = jnp.argmax(scores)
    head_position = valid_positions[head_index]


    # Vérifier le contenu des positions devant (simplifié en 1D)
    push_in_bounds = jnp.all((push_board_positions >= 0) & 
                            (push_board_positions < board.shape[0]), axis=1)
   # #print("push in bounds", push_in_bounds)
    push_contents = board[push_board_positions[:, 0],
                        push_board_positions[:, 1],
                        push_board_positions[:, 2]]
    
    is_opposing = push_contents == -1  # L'adversaire est toujours -1
    has_opposing = is_opposing[0]
    n_opposing = jnp.argmin(is_opposing)

    no_friendly_behind = (n_opposing == 0) | (push_contents[n_opposing] != 1)  # Allié est toujours 1
    can_push = (n_opposing > 0) & (n_opposing < group_size) & no_friendly_behind

    actual_board_state = board[board_positions[:, 0], 
                            board_positions[:, 1], 
                            board_positions[:, 2]]
   


    is_empty = actual_board_state == 0  # La case est vide
    is_moving_piece = (actual_board_state == 1) & positions_mask  # Bille du mouvement

# Valider les positions cibles
    is_moving_piece = positions_mask  # Les positions faisant partie du groupe qui bouge

    is_valid_position = is_empty | is_moving_piece


    # Une fois qu'on a head_position, on regarde la case devant
    head_position = valid_positions[head_index]
    position_ahead = head_position + DIRECTIONS[direction]  # On ajoute le vecteur direction
    position_ahead_board = position_ahead + radius  # On ajoute le radius pour les coordonnées du plateau

    # Vérifier le contenu
    content_ahead = board[position_ahead_board[0], 
                        position_ahead_board[1], 
                        position_ahead_board[2]]


    # Déterminer si une bille alliée bloque le mouvement
    is_blocking_friend_ahead = content_ahead == 1  # On compare avec moving_color plutôt que 1

    destinations_valid = jnp.all(is_valid_position) & ~is_blocking_friend_ahead & ((content_ahead == 0) | (has_opposing & can_push))
    success = is_valid & is_valid_direction & in_bounds & jnp.where(has_opposing,
                                                                can_push,
                                                                destinations_valid)

    # 7. Mise à jour du plateau si le mouvement est valide
        # Créer une version mise à jour pour un mouvement simple
    updated_board = board.copy()

    for i in range(positions.shape[0]):
        # On crée un nouveau board qui aurait la position i mise à 0
        temp_board = updated_board.at[
            start_positions[i, 0],
            start_positions[i, 1],
            start_positions[i, 2]
        ].set(0)
        # On n'applique ce changement que si le masque est True
        updated_board = jnp.where(positions_mask[i], temp_board, updated_board)

    

    # 2. Ensuite placer toutes les billes aux nouvelles positions
    for i in range(positions.shape[0]):
            temp_board = updated_board.at[
                board_positions[i, 0],
                board_positions[i, 1],
                board_positions[i, 2]
            ].set(1)  # Toujours 1 pour nos billes
            updated_board = jnp.where(positions_mask[i], temp_board, updated_board)

    push_updated_board = board.copy()


    for i in range(positions.shape[0]):
        # Préparer le board avec la position vidée
        temp_board = push_updated_board.at[
            start_positions[i, 0],
            start_positions[i, 1],
            start_positions[i, 2]
        ].set(0)
        # Appliquer seulement si le masque est True
        push_updated_board = jnp.where(positions_mask[i], temp_board, push_updated_board)

    # Vider les positions des billes adverses
    for i in range(3):
        temp_board = push_updated_board.at[
            push_board_positions[i, 0],
            push_board_positions[i, 1],
            push_board_positions[i, 2]
        ].set(0)
        push_updated_board = jnp.where(i < n_opposing, temp_board, push_updated_board)

    # 2. Placer toutes les billes aux nouvelles positions
    # D'abord nos billes
    for i in range(positions.shape[0]):
        temp_board = push_updated_board.at[
            board_positions[i, 0],
            board_positions[i, 1],
            board_positions[i, 2]
        ].set(1)
        push_updated_board = jnp.where(positions_mask[i], temp_board, push_updated_board)

    # Puis les billes adverses
    push_dest_positions = push_positions + dir_vec
    push_dest_board_positions = push_dest_positions + radius
    
    for i in range(3):
        temp_board = push_updated_board.at[
            push_dest_board_positions[i, 0],
            push_dest_board_positions[i, 1],
            push_dest_board_positions[i, 2]
        ].set(-1)  # Toujours -1 pour l'adversaire
        push_updated_board = jnp.where(
            (i < n_opposing) & push_in_bounds[i + 1],
            temp_board,
            push_updated_board
        )

    new_board = jnp.where(success,
                        jnp.where(has_opposing, push_updated_board, updated_board),
                        board)

    out_of_bounds = jnp.zeros(3, dtype=bool)  # Initialiser avec False
    out_of_bounds = out_of_bounds.at[:2].set(~push_in_bounds[1:])  # Set les 2 premières positions
    # ##print("out of bounds après modif", out_of_bounds)
    
    # Ne compter que les positions correspondant à des billes adverses et hors limites
    potential_exits = jnp.where(
        jnp.arange(3) < n_opposing,
        out_of_bounds,
        False
    )
    # Appliquer le calcul seulement si c'est une poussée réussie
    billes_sorties = jnp.where(
        has_opposing & success,
        jnp.sum(potential_exits),
        0
    )

    return new_board, success, billes_sorties

# #good one
# @partial(jax.jit, static_argnames=['radius'])
# def move_group_inline(board: chex.Array, 
#                     positions: chex.Array,
#                     direction: int,
#                     group_size: int,
#                     radius: int = 4) -> tuple[chex.Array, bool, int]:
#     """
#     Déplace un groupe de billes dans la direction de leur alignement
#     Gère à la fois les déplacements simples et les poussées
#     Le joueur courant est toujours représenté par 1, l'adversaire par -1
#     """
#     # 1. Préparation des positions et masques
#     positions_mask = jnp.arange(positions.shape[0]) < group_size
#     valid_positions = jnp.where(positions_mask[:, None], positions, 0)
#     dir_vec = DIRECTIONS[direction]
    
#     # 2. Analyse du groupe et validation
#     is_valid, inline_dirs, _ = analyze_group(valid_positions, board, group_size, radius)
#     is_valid_direction = inline_dirs[direction]

#     # 3. Calcul des scores pour trouver la tête du groupe
#     scores = jnp.sum(valid_positions * dir_vec, axis=1)
#     scores = jnp.where(positions_mask, scores, -jnp.inf)
#     head_index = jnp.argmax(scores)
#     head_position = valid_positions[head_index]

#     # 4. Calcul des nouvelles positions
#     start_positions = valid_positions + radius
#     new_positions = jnp.where(positions_mask[:, None], positions + dir_vec, 0)
#     board_positions = new_positions + radius
    
    
#     # 5. Analyse des positions de poussée
#     push_positions = head_position + jnp.array([dir_vec * (i + 1) for i in range(3)])
#     push_board_positions = push_positions + radius
#     push_in_bounds = jnp.all((push_board_positions >= 0) & 
#                             (push_board_positions < board.shape[0]), axis=1)

#     # 6. Vérification des billes adverses
#     push_contents = board[push_board_positions[:, 0],
#                         push_board_positions[:, 1],
#                         push_board_positions[:, 2]]
#     is_opposing = push_contents == -1
#     has_opposing = is_opposing[0]
#     n_opposing = jnp.argmin(is_opposing)
#     no_friendly_behind = (n_opposing == 0) | (push_contents[n_opposing] != 1)
#     can_push = (n_opposing > 0) & (n_opposing < group_size) & no_friendly_behind

#     # 7. Validation des positions cibles

#     actual_board_state = jnp.where(
#     positions_mask,
#     board[board_positions[:, 0],
#           board_positions[:, 1],
#           board_positions[:, 2]],
#     0
# )
    
#     is_empty = actual_board_state == 0

#     is_moving_piece = positions_mask  # Comme avant, mais sans le slice dynamique
#     is_valid_position = is_empty | is_moving_piece


#     # 8. Vérification des conditions de succès
#     in_bounds = jnp.all((board_positions >= 0) & (board_positions < board.shape[0]))
#     destinations_valid = jnp.all(is_valid_position) & ((push_contents[0] == 0) | (has_opposing & can_push))
#     success = is_valid & is_valid_direction & in_bounds & jnp.where(has_opposing,
#                                                                 can_push,
#                                                                 destinations_valid)

#     # 9. Mise à jour du plateau pour mouvement simple

#     start_indices = (
#         jnp.where(positions_mask, start_positions[:, 0], 0),
#         jnp.where(positions_mask, start_positions[:, 1], 0),
#         jnp.where(positions_mask, start_positions[:, 2], 0)
#     )

#     board_indices = (
#     jnp.where(positions_mask, board_positions[:, 0], 0),
#     jnp.where(positions_mask, board_positions[:, 1], 0),
#     jnp.where(positions_mask, board_positions[:, 2], 0)
# )

#     # Enlever les billes des positions de départ
#     updated_board = board.at[start_indices].set(
#         jnp.where(positions_mask, 0., board[start_indices])
#     )
#     # Placer les billes aux nouvelles positions
#     updated_board = updated_board.at[board_indices].set(
#         jnp.where(positions_mask, 1., updated_board[board_indices])
#     )

#     # 10. Mise à jour du plateau pour la poussée
#     push_updated_board = updated_board.copy()
#     push_dest_positions = push_positions + dir_vec
#     push_dest_board_positions = push_dest_positions + radius
#     dest_indices = (push_dest_board_positions[:, 0], 
#                 push_dest_board_positions[:, 1], 
#                 push_dest_board_positions[:, 2])

#     # Création d'un masque statique pour les poussées valides
#     valid_push = jnp.array([True, True, False])  # Toujours taille 3, masqué par n_opposing plus tard
#     valid_push = valid_push & (jnp.arange(3) < n_opposing) & push_in_bounds

#     push_updated_board = push_updated_board.at[dest_indices].set(
#         jnp.where(valid_push, -1, push_updated_board[dest_indices])
#     )

#     # 11. Finalisation
#     new_board = jnp.where(success,
#                         jnp.where(has_opposing, push_updated_board, updated_board),
#                         board)

#     push_possibility = jnp.array([False, True, True])  # positions 1 et 2 sont où les billes peuvent sortir

#     # On combine avec push_in_bounds pour savoir lesquelles sont vraiment hors limite
#     out_of_bounds = push_possibility & ~push_in_bounds

#     # Ne compter que les positions correspondant à des billes adverses et hors limites
#     opposing_mask = jnp.roll(jnp.arange(3) < n_opposing, shift=1)  # donnera [False True False]
#     potential_exits = jnp.where(opposing_mask, out_of_bounds, False)

#     # Appliquer le calcul seulement si c'est une poussée réussie
#     billes_sorties = jnp.where(has_opposing & success, jnp.sum(potential_exits), 0)

#     return new_board, success, billes_sorties

#good one
# @partial(jax.jit, static_argnames=['radius'])
# def move_group_inline(board: chex.Array, 
#                     positions: chex.Array,
#                     direction: int,
#                     group_size: int,
#                     radius: int = 4) -> tuple[chex.Array, bool, int]:
#     """
#     Déplace un groupe de billes dans la direction de leur alignement
#     Gère à la fois les déplacements simples et les poussées
#     Le joueur courant est toujours représenté par 1, l'adversaire par -1
#     """
#     # 1. Préparation des positions et masques (inchangé)
#     positions_mask = jnp.arange(positions.shape[0]) < group_size
#     valid_positions = jnp.where(positions_mask[:, None], positions, 0)
#     dir_vec = DIRECTIONS[direction]
    
#     # 2. Analyse du groupe et validation de base
#     is_valid, inline_dirs, _ = analyze_group(valid_positions, board, group_size, radius)
#     is_valid_direction = inline_dirs[direction]

#     # 3. Calcul de la tête du groupe pour la poussée
#     scores = jnp.sum(valid_positions * dir_vec, axis=1)
#     scores = jnp.where(positions_mask, scores, -jnp.inf)
#     head_index = jnp.argmax(scores)
#     head_position = valid_positions[head_index]

#     # 4. Calcul des nouvelles positions (pour le mouvement simple)
#     start_positions = valid_positions + radius
#     new_positions = jnp.where(positions_mask[:, None], positions + dir_vec, 0)
#     board_positions = new_positions + radius

#     # 5. Vérification initiale des limites et validité
#     in_bounds = jnp.all((board_positions >= 0) & (board_positions < board.shape[0]))
#     actual_board_state = jnp.where(
#         positions_mask,
#         board[board_positions[:, 0], board_positions[:, 1], board_positions[:, 2]],
#         0
#     )
#     is_empty = actual_board_state == 0
#     is_moving_piece = positions_mask
#     is_valid_position = is_empty | is_moving_piece
#     basic_valid = is_valid & is_valid_direction & in_bounds & jnp.all(is_valid_position)

#     # 6. Early check pour la présence de bille adverse
#     push_positions = head_position + jnp.array([dir_vec * (i + 1) for i in range(3)])
#     push_board_positions = push_positions + radius
#     push_contents = board[push_board_positions[:, 0],
#                         push_board_positions[:, 1],
#                         push_board_positions[:, 2]]
#     has_opposing = push_contents[0] == -1

#     # 7. Si succès de base, préparer le mouvement simple
#     start_indices = (
#         jnp.where(positions_mask, start_positions[:, 0], 0),
#         jnp.where(positions_mask, start_positions[:, 1], 0),
#         jnp.where(positions_mask, start_positions[:, 2], 0)
#     )
#     board_indices = (
#         jnp.where(positions_mask, board_positions[:, 0], 0),
#         jnp.where(positions_mask, board_positions[:, 1], 0),
#         jnp.where(positions_mask, board_positions[:, 2], 0)
#     )
    
#     # Mise à jour simple du plateau
#     updated_board = board.at[start_indices].set(
#         jnp.where(positions_mask, 0., board[start_indices])
#     )
#     updated_board = updated_board.at[board_indices].set(
#         jnp.where(positions_mask, 1., updated_board[board_indices])
#     )

#     # 8. Si bille adverse, calculer la validité de la poussée
#     push_in_bounds = jnp.all((push_board_positions >= 0) & 
#                             (push_board_positions < board.shape[0]), axis=1)
#     is_opposing = push_contents == -1
#     n_opposing = jnp.where(has_opposing, 
#                           jnp.argmin(is_opposing),
#                           0)
#     no_friendly_behind = (n_opposing == 0) | (push_contents[n_opposing] != 1)
#     can_push = jnp.where(has_opposing,
#                         (n_opposing > 0) & (n_opposing < group_size) & no_friendly_behind,
#                         False)

#     # 9. Mise à jour finale selon le type de mouvement
#     success = basic_valid & jnp.where(has_opposing, can_push, True)

#     # Calcul des indices de destination pour la poussée
#     push_dest_positions = push_positions + dir_vec
#     push_dest_board_positions = push_dest_positions + radius
#     dest_indices = (push_dest_board_positions[:, 0], 
#                 push_dest_board_positions[:, 1], 
#                 push_dest_board_positions[:, 2])
#     valid_push = jnp.array([True, True, False]) 
#     valid_push = valid_push & (jnp.arange(3) < n_opposing) & push_in_bounds

#     # Si poussée valide, calculer la mise à jour avec poussée
#     push_updated_board = jnp.where(
#         has_opposing & success,
#         # Faire la mise à jour avec poussée
#         updated_board.at[dest_indices].set(
#             jnp.where(valid_push, -1, updated_board[dest_indices])
#         ),
#         # Sinon garder le mouvement simple
#         updated_board
#     )
#     # 10. Calcul des billes sorties seulement si poussée
#     push_possibility = jnp.array([False, True, True])
#     out_of_bounds = push_possibility & ~push_in_bounds
#     opposing_mask = jnp.roll(jnp.arange(3) < n_opposing, shift=1)
#     potential_exits = jnp.where(opposing_mask, out_of_bounds, False)
#     billes_sorties = jnp.where(has_opposing & success, jnp.sum(potential_exits), 0)

#     # 11. Finalisation
#     new_board = jnp.where(success, push_updated_board, board)

#     return new_board, success, billes_sorties
@partial(jax.jit, static_argnames=['radius'])
def move_group_inline(board: chex.Array, 
                    positions: chex.Array,
                    direction: int,
                    group_size: int,
                    radius: int = 4) -> tuple[chex.Array, bool, int]:
    """
    Déplace un groupe de billes dans la direction de leur alignement
    Gère à la fois les déplacements simples et les poussées
    Le joueur courant est toujours représenté par 1, l'adversaire par -1
    """
    # 1. Préparation des positions et masques
    positions_mask = jnp.arange(positions.shape[0]) < group_size
    valid_positions = jnp.where(positions_mask[:, None], positions, 0)
    dir_vec = DIRECTIONS[direction]
    
    # 2. Analyse du groupe et validation de base
    is_valid, inline_dirs, _ = analyze_group(valid_positions, board, group_size, radius)
    is_valid_direction = inline_dirs[direction]

    # 3. Calcul de la tête du groupe pour la poussée
    scores = jnp.sum(valid_positions * dir_vec, axis=1)
    scores = jnp.where(positions_mask, scores, -jnp.inf)
    head_index = jnp.argmax(scores)
    head_position = valid_positions[head_index]

    # 4. Calcul des positions dans l'espace du plateau
    board_space_positions = positions + radius  # positions dans l'espace du plateau
    start_positions = jnp.where(positions_mask[:, None], board_space_positions, 0)
    new_board_positions = jnp.where(positions_mask[:, None], board_space_positions + dir_vec, 0)

    # Pour les positions de poussée
    head_board_position = head_position + radius
    push_board_positions = head_board_position + jnp.array([dir_vec * (i + 1) for i in range(3)])

    # 5. Vérification initiale des limites et validité
    in_bounds = jnp.all((new_board_positions >= 0) & (new_board_positions < board.shape[0]))
    actual_board_state = jnp.where(
        positions_mask,
        board[new_board_positions[:, 0], new_board_positions[:, 1], new_board_positions[:, 2]],
        0
    )
    is_empty = actual_board_state == 0
    is_moving_piece = positions_mask
    is_valid_position = is_empty | is_moving_piece
    basic_valid = is_valid & is_valid_direction & in_bounds & jnp.all(is_valid_position)

    # 6. Early check pour la présence de bille adverse
    push_contents = board[push_board_positions[:, 0],
                        push_board_positions[:, 1],
                        push_board_positions[:, 2]]
    has_opposing = push_contents[0] == -1

    # 7. Si succès de base, préparer le mouvement simple
    # Version vectorisée pour les indices
    masked_start = jnp.where(positions_mask[:, None], start_positions, 0)
    start_indices = (masked_start[:, 0], masked_start[:, 1], masked_start[:, 2])
    
    masked_new = jnp.where(positions_mask[:, None], new_board_positions, 0)
    board_indices = (masked_new[:, 0], masked_new[:, 1], masked_new[:, 2])
    
    # Mise à jour simple du plateau
    updated_board = board.at[start_indices].set(
        jnp.where(positions_mask, 0., board[start_indices])
    )
    updated_board = updated_board.at[board_indices].set(
        jnp.where(positions_mask, 1., updated_board[board_indices])
    )

    # 8. Si bille adverse, calculer la validité de la poussée
    push_in_bounds = jnp.all((push_board_positions >= 0) & 
                            (push_board_positions < board.shape[0]), axis=1)
    is_opposing = push_contents == -1
    n_opposing = jnp.where(has_opposing, 
                          jnp.argmin(is_opposing),
                          0)
    no_friendly_behind = (n_opposing == 0) | (push_contents[n_opposing] != 1)
    can_push = jnp.where(has_opposing,
                        (n_opposing > 0) & (n_opposing < group_size) & no_friendly_behind,
                        False)

    # 9. Mise à jour finale selon le type de mouvement
    success = basic_valid & jnp.where(has_opposing, can_push, True)

    # Calcul des indices de destination pour la poussée
    push_dest_positions = push_board_positions + dir_vec
    dest_indices = (push_dest_positions[:, 0], 
                   push_dest_positions[:, 1], 
                   push_dest_positions[:, 2])
    valid_push = jnp.array([True, True, False]) 
    valid_push = valid_push & (jnp.arange(3) < n_opposing) & push_in_bounds

    # Si poussée valide, calculer la mise à jour avec poussée
    push_updated_board = jnp.where(
        has_opposing & success,
        # Faire la mise à jour avec poussée
        updated_board.at[dest_indices].set(
            jnp.where(valid_push, -1, updated_board[dest_indices])
        ),
        # Sinon garder le mouvement simple
        updated_board
    )

    # 10. Calcul des billes sorties seulement si poussée
    push_possibility = jnp.array([False, True, True])
    out_of_bounds = push_possibility & ~push_in_bounds
    opposing_mask = jnp.roll(jnp.arange(3) < n_opposing, shift=1)
    potential_exits = jnp.where(opposing_mask, out_of_bounds, False)
    billes_sorties = jnp.where(has_opposing & success, jnp.sum(potential_exits), 0)

    # 11. Finalisation
    new_board = jnp.where(success, push_updated_board, board)

    return new_board, success, billes_sorties
# @partial(jax.jit, static_argnames=['radius'])
# def move_group_inline_simple(board: chex.Array, 
#                            positions: chex.Array,
#                            direction: int,
#                            group_size: int,
#                            radius: int = 4) -> tuple[chex.Array, bool, int]:
#     """
#     Déplace un groupe de billes dans la direction de leur alignement
#     Version simplifiée sans poussée
#     """
#     # 1. Préparation des positions et masques
#     positions_mask = jnp.arange(positions.shape[0]) < group_size
#     valid_positions = jnp.where(positions_mask[:, None], positions, 0)
#     dir_vec = DIRECTIONS[direction]
    
#     # 2. Analyse du groupe et validation de base
#     is_valid, inline_dirs, _ = analyze_group(valid_positions, board, group_size, radius)
#     is_valid_direction = inline_dirs[direction]

#     # 3. Calcul des nouvelles positions
#     start_positions = valid_positions + radius
#     new_positions = jnp.where(positions_mask[:, None], positions + dir_vec, 0)
#     board_positions = new_positions + radius

#     # 4. Vérification des limites et validité
#     in_bounds = jnp.all((board_positions >= 0) & (board_positions < board.shape[0]))
#     actual_board_state = jnp.where(
#         positions_mask,
#         board[board_positions[:, 0], board_positions[:, 1], board_positions[:, 2]],
#         0
#     )
#     is_empty = actual_board_state == 0
#     is_moving_piece = positions_mask
#     is_valid_position = is_empty | is_moving_piece
#     success = is_valid & is_valid_direction & in_bounds & jnp.all(is_valid_position)

#     # 5. Mise à jour du plateau si succès
#     start_indices = (
#         jnp.where(positions_mask, start_positions[:, 0], 0),
#         jnp.where(positions_mask, start_positions[:, 1], 0),
#         jnp.where(positions_mask, start_positions[:, 2], 0)
#     )
#     board_indices = (
#         jnp.where(positions_mask, board_positions[:, 0], 0),
#         jnp.where(positions_mask, board_positions[:, 1], 0),
#         jnp.where(positions_mask, board_positions[:, 2], 0)
#     )
    
#     updated_board = board.at[start_indices].set(
#         jnp.where(positions_mask, 0., board[start_indices])
#     )
#     updated_board = updated_board.at[board_indices].set(
#         jnp.where(positions_mask, 1., updated_board[board_indices])
#     )

#     new_board = jnp.where(success, updated_board, board)

#     # Return avec billes_sorties = 0 car pas de poussée
#     return new_board, success, 0

# @partial(jax.jit, static_argnames=['radius'])
# def move_group_inline_push(board: chex.Array, 
#                          positions: chex.Array,
#                          direction: int,
#                          group_size: int,
#                          radius: int = 4) -> tuple[chex.Array, bool, int]:
#     """
#     Déplace un groupe de billes dans la direction de leur alignement avec poussée
#     Version spécialisée pour la gestion des poussées
#     """
#     # 1. Préparation des positions et masques
#     positions_mask = jnp.arange(positions.shape[0]) < group_size
#     valid_positions = jnp.where(positions_mask[:, None], positions, 0)
#     dir_vec = DIRECTIONS[direction]
    
#     # 2. Analyse du groupe et validation de base
#     is_valid, inline_dirs, _ = analyze_group(valid_positions, board, group_size, radius)
#     is_valid_direction = inline_dirs[direction]

#     # 3. Calcul de la tête du groupe pour la poussée
#     scores = jnp.sum(valid_positions * dir_vec, axis=1)
#     scores = jnp.where(positions_mask, scores, -jnp.inf)
#     head_position = valid_positions[jnp.argmax(scores)]

#     # 4. Calcul des positions pour le mouvement de base
#     start_positions = valid_positions + radius
#     new_positions = jnp.where(positions_mask[:, None], positions + dir_vec, 0)
#     board_positions = new_positions + radius

#     # 5. Analyse des positions de poussée
#     push_positions = head_position + jnp.array([dir_vec * (i + 1) for i in range(3)])
#     push_board_positions = push_positions + radius
#     push_in_bounds = jnp.all((push_board_positions >= 0) & 
#                             (push_board_positions < board.shape[0]), axis=1)
    
#     # 6. Analyse de la configuration de poussée
#     push_contents = board[push_board_positions[:, 0],
#                          push_board_positions[:, 1],
#                          push_board_positions[:, 2]]
#     is_opposing = push_contents == -1
#     n_opposing = jnp.argmin(is_opposing)
#     no_friendly_behind = (n_opposing == 0) | (push_contents[n_opposing] != 1)
#     can_push = (n_opposing > 0) & (n_opposing < group_size) & no_friendly_behind

#     # 7. Vérification des limites et validité des positions cibles
#     in_bounds = jnp.all((board_positions >= 0) & (board_positions < board.shape[0]))
#     actual_board_state = jnp.where(
#         positions_mask,
#         board[board_positions[:, 0], board_positions[:, 1], board_positions[:, 2]],
#         0
#     )
#     is_empty = actual_board_state == 0
#     is_moving_piece = positions_mask
#     is_valid_position = is_empty | is_moving_piece
#     basic_valid = is_valid & is_valid_direction & in_bounds & jnp.all(is_valid_position)

#     # 8. Success condition spécifique à la poussée
#     success = basic_valid & can_push

#     # 9. Mise à jour du plateau avec poussée si succès
#     start_indices = (
#         jnp.where(positions_mask, start_positions[:, 0], 0),
#         jnp.where(positions_mask, start_positions[:, 1], 0),
#         jnp.where(positions_mask, start_positions[:, 2], 0)
#     )
#     board_indices = (
#         jnp.where(positions_mask, board_positions[:, 0], 0),
#         jnp.where(positions_mask, board_positions[:, 1], 0),
#         jnp.where(positions_mask, board_positions[:, 2], 0)
#     )

#     # Mise à jour de base
#     updated_board = board.at[start_indices].set(
#         jnp.where(positions_mask, 0., board[start_indices])
#     )
#     updated_board = updated_board.at[board_indices].set(
#         jnp.where(positions_mask, 1., updated_board[board_indices])
#     )

#     # Mise à jour des positions poussées
#     push_dest_positions = push_positions + dir_vec
#     push_dest_board_positions = push_dest_positions + radius
#     dest_indices = (push_dest_board_positions[:, 0],
#                    push_dest_board_positions[:, 1],
#                    push_dest_board_positions[:, 2])
#     valid_push = jnp.array([True, True, False])
#     valid_push = valid_push & (jnp.arange(3) < n_opposing) & push_in_bounds

#     push_updated_board = updated_board.at[dest_indices].set(
#         jnp.where(valid_push, -1, updated_board[dest_indices])
#     )

#     # 10. Calcul des billes sorties
#     push_possibility = jnp.array([False, True, True])
#     out_of_bounds = push_possibility & ~push_in_bounds
#     opposing_mask = jnp.roll(jnp.arange(3) < n_opposing, shift=1)
#     potential_exits = jnp.where(opposing_mask, out_of_bounds, False)
#     billes_sorties = jnp.where(success, jnp.sum(potential_exits), 0)

#     # 11. Finalisation
#     new_board = jnp.where(success, push_updated_board, board)

#     return new_board, success, billes_sorties

# @partial(jax.jit, static_argnames=['radius'])
# def move_group_inline(board: chex.Array, 
#                     positions: chex.Array,
#                     direction: int,
#                     group_size: int,
#                     radius: int = 4) -> tuple[chex.Array, bool, int]:
#     """
#     Déplace un groupe de billes dans la direction de leur alignement
#     Dispatch vers la version avec ou sans poussée selon la présence d'une bille adverse
#     """
#     # 1. Check rapide pour la présence d'une bille adverse
#     dir_vec = DIRECTIONS[direction]
    
#     # Trouver la tête du groupe
#     positions_mask = jnp.arange(positions.shape[0]) < group_size
#     valid_positions = jnp.where(positions_mask[:, None], positions, 0)
#     scores = jnp.sum(valid_positions * dir_vec, axis=1)
#     scores = jnp.where(positions_mask, scores, -jnp.inf)
#     head_position = valid_positions[jnp.argmax(scores)]
    
#     # Vérifier la présence d'une bille adverse
#     push_position = head_position + dir_vec
#     push_board_position = push_position + radius
#     cell_content = board[push_board_position[0], 
#                         push_board_position[1],
#                         push_board_position[2]]
    
#     return jax.lax.switch(
#         jnp.where(cell_content == -1, 1, 0),  # index: 0 pour simple, 1 pour push
#         [
#             lambda x: move_group_inline_simple(*x),
#             lambda x: move_group_inline_push(*x)
#         ],
#         (board, positions, direction, group_size, radius)
#     )