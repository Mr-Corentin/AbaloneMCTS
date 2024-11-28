import jax.numpy as jnp
import pytest

from src.jax_game import JaxAbaloneGame
from src.jax_utils import get_cell_content_jax

@pytest.fixture
def create_test_board():
    def _create_test_board(game, marble_positions):
        board = game.board
        for y in range(9):
            for x in range(9):
                if get_cell_content_jax(board, x, y) in ['W', 'B']:
                    board = board.at[y, x].set(0)
        for color in ['W', 'B']:
            if color in marble_positions:
                for x, y in marble_positions[color]:
                    value = 1 if color == 'W' else -1
                    board = board.at[y, x].set(value)
        # Mettre à jour directement l'attribut board
        game.board = board
        return game
    return _create_test_board

@pytest.mark.parametrize(
    "initial_positions, move, expected_whites, expected_black, should_succeed, error_message",
    [
        (
            {'W': [(3, 3), (4, 3), (5, 3)], 'B': [(6, 3)]},
            ([(3, 3), (4, 3), (5, 3)], 'E'),
            [(4, 3), (5, 3), (6, 3)],
            [(7, 3)],
            True,
            None
        ),
        (
            {'W': [(4, 3), (5, 3)], 'B': [(6, 3)]},
            ([(4, 3), (5, 3)], 'E'),
            [(5, 3), (6, 3)],
            [(7, 3)],
            True,
            None
        ),
        (
            {'W': [(1, 3), (2, 3), (3, 3), (4, 3)], 'B': [(5, 3), (6, 3)]},
            ([(1, 3), (2, 3), (3, 3), (4, 3)], 'E'),
            [(1, 3), (2, 3), (3, 3), (4, 3)],
            [(5, 3), (6, 3)],
            False,
            "mouvement invalide"
        ),
        (
            {'W': [(1, 0)], 'B': [(2, 1), (3, 2), (4, 3)]},
            ([(2, 1), (3, 2), (4, 3)], 'NW'),
            [(1, 0)],
            [(2, 1), (3, 2), (4, 3)],
            False,
            "ces billes ne vous appartiennent pas"
        )
    ]
)
def test_push_cases(create_test_board, initial_positions, move, expected_whites, expected_black, should_succeed, error_message):
    game = JaxAbaloneGame()
    game = create_test_board(game, initial_positions)
    game.current_player = 1 if 'W' in initial_positions else -1

    success, message = game.make_move(*move)

    assert success == should_succeed, f"Expected success={should_succeed}, got {success}. Message: {message}"
    if not should_succeed and error_message:
        assert error_message in message.lower(), f"Expected error '{error_message}' in message '{message}'"

    for x, y in expected_whites:
        assert get_cell_content_jax(game.board, x, y) == 1, f"Expected W at ({x}, {y})"
    for x, y in expected_black:
        assert get_cell_content_jax(game.board, x, y) == -1, f"Expected B at ({x}, {y})"

def test_push_out_of_board(create_test_board):
    game = JaxAbaloneGame()
    positions = {'W': [(5, 5), (6, 5)], 'B': [(7, 5)]}
    game = create_test_board(game, positions)

    initial_score = game.black_marbles_out
    success, message = game.make_move([(5, 5), (6, 5)], 'E')

    assert success, f"Move failed: {message}"
    expected_whites = [(6, 5), (7, 5)]

    for x, y in expected_whites:
        assert get_cell_content_jax(game.board, x, y) == 1, f"Expected W at ({x}, {y})"
    assert game.black_marbles_out == initial_score + 1, f"Expected black marble count to increase"


def test_win_condition(create_test_board):
    game = JaxAbaloneGame()
    game.white_marbles_out = 5  # Mise à jour directe
    positions = {'W': [(0, 2)], 'B': [(2, 4), (1, 3)]}
    game = create_test_board(game, positions)
    game.switch_player()
    success, message = game.make_move([(2, 4), (1, 3)], 'NW')

    assert success, f"Move failed: {message}"
    assert game.white_marbles_out == 6, "Expected 6 white marbles to be out"
    is_over, win_message = game.is_game_over()
    assert is_over, f"Expected the game to be over"
    assert "Noirs ont gagné" in win_message, f"Unexpected win message: {win_message}"

