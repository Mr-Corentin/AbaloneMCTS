import pytest
from src.game import AbaloneGame
from src.utils import get_cell_content

@pytest.fixture
def create_test_board():
    def _create_test_board(game, marble_positions):
        for y in range(9):
            for x in range(9):
                if get_cell_content(game.board, x, y) in ['W', 'B']:
                    game.board[y][x] = 0
        for color in ['W', 'B']:
            if color in marble_positions:
                for x, y in marble_positions[color]:
                    game.board[y][x] = 1 if color == 'W' else -1
    return _create_test_board
@pytest.mark.parametrize(
    "initial_positions, move, expected_whites, expected_black, should_succeed, error_message",
    [
        # Cas 1 : 3 blanches poussent 1 noire
        (
            {'W': [(3, 3), (4, 3), (5, 3)], 'B': [(6, 3)]},
            ([(3, 3), (4, 3), (5, 3)], 'E'),
            [(4, 3), (5, 3), (6, 3)],
            [(7, 3)],
            True,
            None
        ),
        # Cas 2 : 2 blanches poussent 1 noire
        (
            {'W': [(4, 3), (5, 3)], 'B': [(6, 3)]},
            ([(4, 3), (5, 3)], 'E'),
            [(5, 3), (6, 3)],
            [(7, 3)],
            True,
            None
        ),
        # Cas 4 : 4 blanches tentent de pousser 2 noires
        (
            {'W': [(1, 3), (2, 3), (3, 3), (4, 3)], 'B': [(5, 3), (6, 3)]},
            ([(1, 3), (2, 3), (3, 3), (4, 3)], 'E'),
            [(1, 3), (2, 3), (3, 3), (4, 3)],
            [(5, 3), (6, 3)],
            False,
            "mouvement invalide"
        ),
        # Cas 5 : 3 noires tentent de pousser 1 blanche
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
    game = AbaloneGame()
    create_test_board(game, initial_positions)
    game.current_player = 'W' if 'W' in initial_positions else 'B'

    success, message = game.make_move(*move)
    assert success == should_succeed, f"Expected success={should_succeed}, got {success}. Message: {message}"
    if not should_succeed and error_message:
        assert error_message in message.lower(), f"Expected error '{error_message}' in message '{message}'"

    for x, y in expected_whites:
        assert get_cell_content(game.board, x, y) == 'W', f"Expected W at ({x}, {y})"
    for x, y in expected_black:
        assert get_cell_content(game.board, x, y) == 'B', f"Expected B at ({x}, {y})"


def test_push_out_of_board(create_test_board):
    game = AbaloneGame()
    positions = {
        'W': [(5, 5), (6, 5)],
        'B': [(7, 5)]
    }
    create_test_board(game, positions)
    game.current_player = 'W'

    initial_score = game.black_marbles_out
    success, message = game.make_move([(5, 5), (6, 5)], 'E')
    assert success, f"Move failed: {message}"

    expected_whites = [(6, 5), (7, 5)]
    for x, y in expected_whites:
        assert get_cell_content(game.board, x, y) == 'W', f"Expected W at ({x}, {y})"

    assert game.black_marbles_out == initial_score + 1, "Expected black marble count to increase"

def test_win_condition(create_test_board):
    game = AbaloneGame()
    game.white_marbles_out = 5
    positions = {
        'W': [(0, 2)],
        'B': [(2, 4), (1, 3)]
    }
    create_test_board(game, positions)
    game.current_player = 'B'
    success, message = game.make_move([(2, 4), (1, 3)], 'NW')
    assert success, f"Move failed: {message}"

    assert game.white_marbles_out == 6, "Expected 6 white marbles to be out"
    is_over, win_message = game.is_game_over()
    assert is_over, "Expected the game to be over"
    assert "Noirs ont gagné" in win_message, f"Unexpected win message: {win_message}"


@pytest.mark.parametrize(
    "initial_positions, move, expected_positions",
    [
        # Cas 1: NW
        ({'W': [(3, 1), (3, 2)], 'B': []}, 'NW', [(2, 0), (2, 1)]),
        # Cas 2: E
        ({'W': [(3, 1), (3, 2)], 'B': []}, 'E', [(4, 1), (4, 2)]),
        # Cas 3: SE
        ({'W': [(3, 1), (3, 2)], 'B': []}, 'SE', [(4, 2), (4, 3)]),
        # Cas 4: W
        ({'W': [(3, 1), (3, 2)], 'B': []}, 'W', [(2, 1), (2, 2)]),
        # Cas 5: NE
        ({'W': [(3, 1), (3, 2)], 'B': []}, 'NE', [(3, 0), (3, 1)]),
        # Cas 6: SW
        ({'W': [(3, 1), (3, 2)], 'B': []}, 'SW', [(3, 2), (3, 3)])
    ]
)
def test_para_upper_moves(create_test_board, initial_positions, move, expected_positions):
    game = AbaloneGame()
    create_test_board(game, initial_positions)
    game.current_player = 'W'

    success, message = game.make_move([(3, 1), (3, 2)], move)
    assert success, f"Move failed: {message}"

    for x, y in expected_positions:
        assert get_cell_content(game.board, x, y) == 'W', f"Expected W at ({x}, {y})"


@pytest.mark.parametrize(
    "initial_positions, move, expected_positions",
    [
        # Cas 1: NW
        ({'W': [(3, 6), (3, 7)], 'B': []}, 'NW', [(3, 5), (3, 6)]),
        # Cas 2: E
        ({'W': [(3, 6), (3, 7)], 'B': []}, 'E', [(4, 6), (4, 7)]),
        # Cas 3: SE
        ({'W': [(3, 6), (3, 7)], 'B': []}, 'SE', [(3, 7), (3, 8)]),
        # Cas 4: W
        ({'W': [(3, 6), (3, 7)], 'B': []}, 'W', [(2, 6), (2, 7)]),
        # Cas 5: NE
        ({'W': [(3, 6), (3, 7)], 'B': []}, 'NE', [(4, 5), (4, 6)]),
        # Cas 6: SW
        ({'W': [(3, 6), (3, 7)], 'B': []}, 'SW', [(2, 7), (2, 8)])
    ]
)
def test_para_lower_moves(create_test_board, initial_positions, move, expected_positions):
    game = AbaloneGame()
    create_test_board(game, initial_positions)
    game.current_player = 'W'

    success, message = game.make_move([(3, 6), (3, 7)], move)
    assert success, f"Move failed: {message}"

    for x, y in expected_positions:
        assert get_cell_content(game.board, x, y) == 'W', f"Expected W at ({x}, {y})"

@pytest.mark.parametrize(
    "initial_positions, move, expected_positions",
    [
        # Cas 1: NW
        ({'W': [(4, 4), (5, 4)], 'B': []}, 'NW', [(3, 3), (4, 3)]),
        # Cas 2: E
        ({'W': [(4, 4), (5, 4)], 'B': []}, 'E', [(5, 4), (6, 4)]),
        # Cas 3: SE
        ({'W': [(4, 4), (5, 4)], 'B': []}, 'SE', [(4, 5), (5, 5)]),
        # Cas 4: W
        ({'W': [(4, 4), (5, 4)], 'B': []}, 'W', [(3, 4), (4, 4)]),
        # Cas 5: NE
        ({'W': [(4, 4), (5, 4)], 'B': []}, 'NE', [(4, 3), (5, 3)]),
        # Cas 6: SW
        ({'W': [(4, 4), (5, 4)], 'B': []}, 'SW', [(3, 5), (4, 5)])
    ]
)
def test_para_mid_moves(create_test_board, initial_positions, move, expected_positions):
    game = AbaloneGame()
    create_test_board(game, initial_positions)
    game.current_player = 'W'

    # Effectuer le mouvement
    success, message = game.make_move([(4, 4), (5, 4)], move)
    assert success, f"Move failed: {message}"

    # Vérifier les positions finales
    for x, y in expected_positions:
        assert get_cell_content(game.board, x, y) == 'W', f"Expected W at ({x}, {y}), but found {get_cell_content(game.board, x, y)}"

@pytest.mark.parametrize(
    "initial_positions, move, expected_positions",
    [
        # Cas 1: NW
        ({'W': [(3, 5), (4, 4), (4, 3)], 'B': []}, 'NW', [(3, 4), (3, 3), (3, 2)]),
        # Cas 2: E
        ({'W': [(3, 5), (4, 4), (4, 3)], 'B': []}, 'E', [(4, 5), (5, 4), (5, 3)]),
        # Cas 3: SE
        ({'W': [(3, 5), (4, 4), (4, 3)], 'B': []}, 'SE', [(3, 6), (4, 5), (5, 4)]),
        # Cas 4: W
        ({'W': [(3, 5), (4, 4), (4, 3)], 'B': []}, 'W', [(2, 5), (3, 4), (3, 3)]),
        # Cas 5: NE
        ({'W': [(3, 5), (4, 4), (4, 3)], 'B': []}, 'NE', [(4, 2), (4, 3), (4, 4)]),
        # Cas 6: SW
        ({'W': [(3, 5), (4, 4), (4, 3)], 'B': []}, 'SW', [(2, 6), (3, 5), (4, 4)])
    ]
)
def test_verti_mid_moves(create_test_board, initial_positions, move, expected_positions):
    game = AbaloneGame()
    create_test_board(game, initial_positions)
    game.current_player = 'W'

    # Effectuer le mouvement
    success, message = game.make_move([(3, 5), (4, 4), (4, 3)], move)
    assert success, f"Move failed: {message}"

    # Vérifier les positions finales
    for x, y in expected_positions:
        assert get_cell_content(game.board, x, y) == 'W', f"Expected W at ({x}, {y}), but found {get_cell_content(game.board, x, y)}"
@pytest.mark.parametrize(
    "initial_positions, move, expected_positions",
    [
        # Cas 1: NW
        ({'W': [(1, 5), (2, 5), (3, 5)], 'B': []}, 'NW', [(1, 4), (2, 4), (3, 4)]),
        # Cas 2: E
        ({'W': [(1, 5), (2, 5), (3, 5)], 'B': []}, 'E', [(2, 5), (3, 5), (4, 5)]),
        # Cas 3: SE
        ({'W': [(1, 5), (2, 5), (3, 5)], 'B': []}, 'SE', [(1, 6), (2, 6), (3, 6)]),
        # Cas 4: W
        ({'W': [(1, 5), (2, 5), (3, 5)], 'B': []}, 'W', [(0, 5), (1, 5), (2, 5)]),
        # Cas 5: NE
        ({'W': [(1, 5), (2, 5), (3, 5)], 'B': []}, 'NE', [(2, 4), (3, 4), (4, 4)]),
        # Cas 6: SW
        ({'W': [(1, 5), (2, 5), (3, 5)], 'B': []}, 'SW', [(0, 6), (1, 6), (2, 6)])
    ]
)
def test_hori_bot_moves(create_test_board, initial_positions, move, expected_positions):
    game = AbaloneGame()
    create_test_board(game, initial_positions)
    game.current_player = 'W'

    # Effectuer le mouvement
    success, message = game.make_move([(1, 5), (2, 5), (3, 5)], move)
    assert success, f"Move failed: {message}"

    # Vérifier les positions finales
    for x, y in expected_positions:
        assert get_cell_content(game.board, x, y) == 'W', f"Expected W at ({x}, {y}), but found {get_cell_content(game.board, x, y)}"
