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
        jnp.where(success, 1, dest_value)  
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
   # ##print("new_pos",new_positions)
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
   # ##print("push in bounds", push_in_bounds)
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
    # ###print("out of bounds après modif", out_of_bounds)
    
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