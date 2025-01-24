import jax
from jax import profiler
import jax.numpy as jnp
from env import AbaloneEnv, AbaloneState
from mcts import run_search, AbaloneMCTSRecurrentFn

from jax.profiler import trace, TraceAnnotation

def profile_game(env, model, params, num_simulations=20, max_moves=10, save_path="./tensorboard"):
    """Profile une partie complète"""
    with trace(save_path):  # Trace principale
        with TraceAnnotation("game_initialization"):
            state = env.reset()
            move_count = 0
        
        with TraceAnnotation("full_game_loop"):
            while not env.is_terminal(state) and move_count < max_moves:
                move_count += 1
                print(f"\n=== Coup n°{move_count} sur {max_moves} ===")
                
                with TraceAnnotation(f"move_{move_count}"):
                    with TraceAnnotation("mcts_search"):
                        policy_output = run_search(
                            state=state,
                            recurrent_fn=AbaloneMCTSRecurrentFn(env, model),
                            network=model,
                            params=params,
                            rng_key=jax.random.PRNGKey(0),
                            env=env,
                            num_simulations=num_simulations
                        )
                        policy_output.action.block_until_ready()
                    
                    with TraceAnnotation("move_execution"):
                        action = policy_output.action
                        state = env.step(state, action)
                        state.board.block_until_ready()
if __name__ == "__main__":
    import jax.random as random
    from network import DummyAbaloneModel

    # Initialisation
    env = AbaloneEnv()
    dummy_model = DummyAbaloneModel()
    rng_key = random.PRNGKey(0)
    
    # Initialisation des paramètres
    dummy_board = jnp.zeros((1, 9, 9))
    dummy_marbles = jnp.zeros((1, 2))
    params = dummy_model.init(rng_key, dummy_board, dummy_marbles)

    # Lancer le profiling
    profile_game(env, dummy_model, params, num_simulations=10)