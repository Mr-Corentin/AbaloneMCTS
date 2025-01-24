import jax
import jax.numpy as jnp
import chex
from typing import Tuple, Dict, NamedTuple
from board import initialize_board
from legal_moves_cano import get_legal_moves
from moves_cano import move_group_inline, move_group_parallel, move_single_marble
import numpy as np
from functools import partial

class AbaloneState(NamedTuple):
    """État du jeu d'Abalone (version canonique)"""
    board: chex.Array  # Le plateau où le joueur courant est toujours 1
    actual_player: int  # Le joueur réel (1=noir, -1=blanc)
    black_out: int  # Nombre de billes noires sorties
    white_out: int  # Nombre de billes blanches sorties
    moves_count: int

class AbaloneEnv:
    def __init__(self, radius: int = 4):
        self.radius = radius
        self.moves_index = self._load_moves_index()

    def reset(self) -> AbaloneState:
        """Réinitialise l'environnement"""
        board = initialize_board()  # Les noirs (1) commencent toujours
        return AbaloneState(
            board=board,
            actual_player=1,  # Noirs commencent
            black_out=0,
            white_out=0,
            moves_count=0
        )


    @partial(jax.jit, static_argnames=['self'])
    def step(self, state: AbaloneState, move_idx: int) -> AbaloneState:
        """Effectue un mouvement et retourne le nouvel état"""
        # Convertir move_idx en entier scalaire
        move_idx = move_idx.astype(jnp.int32).reshape(())
        
        # Accéder aux données avec JAX
        positions = jnp.array(self.moves_index['positions'])
        direction = jnp.array(self.moves_index['directions'])[move_idx]
        move_type = jnp.array(self.moves_index['move_types'])[move_idx]
        group_size = jnp.array(self.moves_index['group_sizes'])[move_idx]
        position = positions[move_idx]

        def single_marble_case(inputs):
            state, position, direction = inputs
            new_board, _ = move_single_marble(state.board, position[0], direction, self.radius)
            return new_board, 0
        
        def group_parallel_case(inputs):
            state, position, direction, group_size = inputs
            new_board, _ = move_group_parallel(state.board, position, direction, group_size, self.radius)
            return new_board, 0
            
        def group_inline_case(inputs):
            state, position, direction, group_size = inputs
            new_board, _, billes_sorties = move_group_inline(state.board, position, direction, group_size, self.radius)
            return new_board, billes_sorties

        # Utiliser switch pour le type de mouvement
        new_board, billes_sorties = jax.lax.switch(
            move_type,
            [
                lambda x: single_marble_case((state, position, direction)),
                lambda x: group_parallel_case((state, position, direction, group_size)),
                lambda x: group_inline_case((state, position, direction, group_size))
            ],
            0
        )

        # S'assurer que actual_player est un scalaire
        actual_player = state.actual_player.reshape(())

        # Mise à jour des billes sorties
        black_out = state.black_out + billes_sorties * (actual_player == -1)
        white_out = state.white_out + billes_sorties * (actual_player == 1)

        return AbaloneState(
            board=-new_board,
            actual_player=-actual_player,
            black_out=black_out,
            white_out=white_out,
            moves_count=state.moves_count + 1
        )
    def _load_moves_index(self):
        """Charge l'index des mouvements à partir du fichier npz"""
        moves_data = np.load('move_map.npz')
        return {
            'positions': moves_data['positions'],
            'directions': moves_data['directions'],
            'move_types': moves_data['move_types'],
            'group_sizes': moves_data['group_sizes']
        }

    def get_legal_moves(self, state: AbaloneState) -> chex.Array:
        """Retourne un masque des mouvements légaux"""
        return get_legal_moves(state.board, self.moves_index, self.radius)

    def is_terminal(self, state: AbaloneState) -> bool:
        """Vérifie si l'état est terminal"""
        # Une partie est terminée si :
        # - 6 billes d'une couleur sont sorties
        # - Ou si on atteint un nombre maximum de coups (ex: 300)
        return (state.black_out >= 6) or (state.white_out >= 6) or (state.moves_count >= 300)

    def get_winner(self, state: AbaloneState) -> int:
        """
        Détermine le gagnant

        Returns:
            1 si les noirs gagnent, -1 si les blancs gagnent, 0 si match nul
        """
        if state.white_out >= 6:
            return 1  # Noirs gagnent
        elif state.black_out >= 6:
            return -1  # Blancs gagnent
        elif state.moves_count >= 300:
            return 0  # Match nul
        return 0  # Partie en cours

    def is_legal_move(self, state: AbaloneState, move_idx: int) -> bool:
        """Vérifie si un mouvement spécifique est légal"""
        legal_moves = self.get_legal_moves(state)
        return legal_moves[move_idx]

    def get_score(self, state: AbaloneState) -> dict:
        """Retourne le score actuel sous forme de dictionnaire"""
        return {
            'black_out': state.black_out,
            'white_out': state.white_out,
            'moves': state.moves_count
        }

    def get_canonical_state(self, board: chex.Array, actual_player: int) -> chex.Array:
        """
        Convertit un plateau en sa représentation canonique où le joueur à jouer est toujours 1

        Args:
            board: État du plateau
            actual_player: Joueur qui doit jouer (1 ou -1)

        Returns:
            board_canonical: Plateau en représentation canonique
        """
        return jnp.where(actual_player == 1, board, -board)

