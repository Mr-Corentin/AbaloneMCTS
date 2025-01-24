import jax
import jax.numpy as jnp
import mctx
from mcts import get_root_output, AbaloneMCTSRecurrentFn

def state_to_arrays(state):
    """Convertit un AbaloneState en arrays JAX"""
    return {
        'board': state.board,
        'actual_player': state.actual_player,
        'black_out': state.black_out,
        'white_out': state.white_out,
        'moves_count': state.moves_count
    }

def arrays_to_state(arrays):
    """Convertit les arrays JAX en AbaloneState"""
    from env import AbaloneState  # Import ici pour éviter les imports circulaires
    return AbaloneState(
        board=arrays['board'],
        actual_player=arrays['actual_player'],
        black_out=arrays['black_out'],
        white_out=arrays['white_out'],
        moves_count=arrays['moves_count']
    )

def parallel_dummy_search(state, dummy_model, params, rng_key, env, num_simulations=300):
    n_devices = 8
    rng_keys = jax.random.split(rng_key, n_devices)
    
    # Convertir l'état en dictionnaire d'arrays
    state_arrays = state_to_arrays(state)
    
    # Répliquer chaque composant de l'état
    sharded_state_arrays = {
        key: jnp.broadcast_to(val, (n_devices,) + val.shape)
        for key, val in state_arrays.items()
    }
    
    @jax.pmap
    def sharded_search(state_arrays, params, rng_key):
        # Reconvertir en AbaloneState pour l'utiliser
        state = arrays_to_state(state_arrays)
        root = get_root_output(state, dummy_model, params, env)
        invalid_actions = jnp.expand_dims(~env.get_legal_moves(state), 0)
        
        return mctx.gumbel_muzero_policy(
            params=params,
            rng_key=rng_key,
            root=root,
            recurrent_fn=AbaloneMCTSRecurrentFn(env, dummy_model).recurrent_fn,
            num_simulations=num_simulations,
            max_num_considered_actions=16,
            invalid_actions=invalid_actions,
            gumbel_scale=0.0
        )
    
    sharded_params = jax.device_put_replicated(params, jax.devices())
    
    return sharded_search(sharded_state_arrays, sharded_params, rng_keys)