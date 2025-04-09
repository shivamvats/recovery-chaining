from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv


def reset_venv(env, states, indices=None, **kwargs):
    """
    Reset a vectorized environment with given states.
    """

    if isinstance(env.unwrapped, SubprocVecEnv):
        target_remotes = env._get_target_remotes(indices)
        for remote, state in zip(target_remotes, states):
            remote.send(("env_method", ("reset_env", (state,), kwargs)))
        return [remote.recv() for remote in target_remotes]

    elif isinstance(env.unwrapped, DummyVecEnv):
        target_envs = env._get_target_envs(indices)
        return [
            getattr(env_i, "reset_env")(state, **kwargs)
            for env_i, state in zip(target_envs, states)
        ]

    else:
        raise NotImplementedError


def venv_method(env, method_name, list_of_args, indices=None, **kwargs):
    """
    Call a method of a vectorized environment.
    """

    if isinstance(env.unwrapped, SubprocVecEnv):
        target_remotes = env._get_target_remotes(indices)
        for remote, args in zip(target_remotes, list_of_args):
            remote.send(("env_method", (method_name, args, kwargs)))
        return [remote.recv() for remote in target_remotes]

    elif isinstance(env.unwrapped, DummyVecEnv):
        target_envs = env._get_target_envs(indices)
        return [
            getattr(env_i, method_name)(*args, **kwargs)
            for env_i, args in zip(target_envs, list_of_args)
        ]
    else:
        raise NotImplementedError
