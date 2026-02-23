from typing import Any, cast

import gymnasium as gym
import numpy as np
from gymnasium.vector import SyncVectorEnv
from ns_gym.schedulers import (
    ContinuousScheduler,
    DiscreteScheduler,
    MemorylessScheduler,
    PeriodicScheduler,
    RandomScheduler,
)
from ns_gym.update_functions import (
    DeterministicTrend,
    ExponentialDecay,
    GeometricProgression,
    IncrementUpdate,
    NoUpdate,
    OscillatingUpdate,
    RandomWalk,
    RandomWalkWithDrift,
    RandomWalkWithDriftAndTrend,
    StepWiseUpdate,
)
from ns_gym.wrappers import NSClassicControlWrapper


# Concrete scheduler types we instantiate; used for runtime check before passing to update functions
schedulerTypes = (
    ContinuousScheduler,
    DiscreteScheduler,
    MemorylessScheduler,
    PeriodicScheduler,
    RandomScheduler,
)


class NSClassicControlWrapperPersistentRNG(NSClassicControlWrapper):
    """NSClassicControlWrapper that keeps the same update functions (and their RNG state) across resets."""

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        persistent_params = self.tunable_params
        obs, info = super().reset(seed=seed, options=options)
        self.tunable_params = persistent_params
        return obs, info


def create_ns_gym_update_functions(ns_gym_config: dict[str, Any]) -> dict[str, Any]:
    """Build param update functions for ns_gym from config."""
    param_update_functions = {}
    for param_name, update_config in ns_gym_config.items():
        scheduler_type = update_config["scheduler"]
        update_func_type = update_config["update_function"]
        scheduler_kwargs = dict(update_config.get("scheduler_kwargs", {}) or {})
        update_kwargs = dict(update_config.get("update_kwargs", {}) or {})
        if scheduler_type == "continuous":
            scheduler = ContinuousScheduler(**scheduler_kwargs)
        elif scheduler_type == "periodic":
            scheduler = PeriodicScheduler(**scheduler_kwargs)
        elif scheduler_type == "random":
            scheduler = RandomScheduler(**scheduler_kwargs)
        elif scheduler_type == "discrete":
            event_list = scheduler_kwargs.pop("event_list", [])
            scheduler = DiscreteScheduler(set(event_list), **scheduler_kwargs)
        elif scheduler_type == "memoryless":
            scheduler = MemorylessScheduler(**scheduler_kwargs)
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")
        if not isinstance(scheduler, schedulerTypes):
            raise TypeError(
                f"Expected a scheduler instance (one of {[t.__name__ for t in schedulerTypes]}), "
                f"got {type(scheduler).__name__}"
            )
        # ns_gym type stubs say scheduler is type[Scheduler]; runtime expects an instance
        sched: Any = cast(Any, scheduler)
        if update_func_type == "increment":
            update_func = IncrementUpdate(sched, **update_kwargs)
        elif update_func_type == "random_walk":
            update_func = RandomWalk(sched, **update_kwargs)
        elif update_func_type == "no_update":
            update_func = NoUpdate(sched)
        elif update_func_type == "deterministic_trend":
            update_func = DeterministicTrend(sched, **update_kwargs)
        elif update_func_type == "exponential_decay":
            update_func = ExponentialDecay(sched, **update_kwargs)
        elif update_func_type == "geometric_progression":
            update_func = GeometricProgression(sched, **update_kwargs)
        elif update_func_type == "oscillating":
            update_func = OscillatingUpdate(sched, **update_kwargs)
        elif update_func_type == "random_walk_with_drift":
            update_func = RandomWalkWithDrift(sched, **update_kwargs)
        elif update_func_type == "random_walk_with_drift_and_trend":
            update_func = RandomWalkWithDriftAndTrend(sched, **update_kwargs)
        elif update_func_type == "step_wise":
            update_func = StepWiseUpdate(sched, **update_kwargs)
        else:
            raise ValueError(f"Unknown update function type: {update_func_type}")
        param_update_functions[param_name] = update_func
    return param_update_functions


def make_ns_env(
    env_id: str,
    non_stationary_params: dict[str, Any],
    max_steps_per_episode: int | None = None,
    env_kwargs: dict[str, Any] | None = None,
    render: bool = False,
) -> gym.Env:
    """Create one non-stationary environment with persistent RNG update functions."""
    env_kwargs = env_kwargs or {}
    if render and "render_mode" not in env_kwargs:
        env_kwargs = {**env_kwargs, "render_mode": "human"}

    param_update_functions = create_ns_gym_update_functions(non_stationary_params)
    base_env = gym.make(
        env_id,
        max_episode_steps=max_steps_per_episode,
        **env_kwargs,
    )
    return NSClassicControlWrapperPersistentRNG(
        base_env,
        param_update_functions,
        change_notification=True,
        delta_change_notification=True,
    )


def validate_supported_env(env_id: str) -> None:
    """Ensure env_id is one of the currently supported classic-control envs."""
    supported_envs = {"CartPole-v1", "MountainCar-v0", "Acrobot-v1", "Pendulum-v1"}
    if env_id not in supported_envs:
        raise ValueError(
            f"Environment {env_id} is not supported for non-stationary learning."
        )


class NSVectorEnvRunner:
    """Persistent non-stationary vector env runner with step state."""

    def __init__(
        self,
        env_id: str,
        non_stationary_params: dict[str, Any],
        seed: int,
        num_envs: int = 1,
        max_steps_per_episode: int | None = None,
        env_kwargs: dict[str, Any] | None = None,
        render: bool = False,
    ):
        validate_supported_env(env_id)
        if num_envs < 1:
            raise ValueError(f"num_envs must be >= 1, got {num_envs}.")

        self.num_envs = num_envs
        self.render = render

        env_kwargs = env_kwargs or {}

        def make_env():
            return make_ns_env(
                env_id=env_id,
                non_stationary_params=non_stationary_params,
                max_steps_per_episode=max_steps_per_episode,
                env_kwargs=env_kwargs,
                render=render,
            )

        env_fns = [make_env for _ in range(num_envs)]
        self.env = SyncVectorEnv(
            env_fns,
            copy=True,
            observation_mode="same",
            autoreset_mode=gym.vector.AutoresetMode.NEXT_STEP,
        )

        self.action_space = self.env.action_space
        self.single_action_space = self.env.single_action_space
        if isinstance(self.single_action_space, gym.spaces.Box):
            raise ValueError("Only discrete action spaces are supported.")
        self.action_dim = int(self.single_action_space.n)

        self.init = True
        self.env_seed = seed

    def sample_random_actions(self) -> np.ndarray:
        """Sample random discrete actions for each env."""
        actions = self.action_space.sample()
        return np.asarray(actions, dtype=np.int64)

    def step(self, actions: np.ndarray | None = None) -> None:
        """Advance env by one step and update state."""
        if self.init:
            self.init = False
            self.actions = self.sample_random_actions()
            obs, self.infos = self.env.reset(seed=self.env_seed)
            self.obs = obs["state"]
            self.rewards = np.zeros((self.num_envs,), dtype=np.float64)
            self.dones = np.zeros((self.num_envs,), dtype=np.bool_)
        else:
            if actions is None:
                self.actions = self.sample_random_actions()
            else:
                self.actions = np.asarray(actions, dtype=np.int64)
            obs, rewards, terminations, truncations, self.infos = self.env.step(self.actions)
            self.obs = obs["state"]
            self.rewards = np.asarray(rewards, dtype=np.float64)
            self.dones = np.logical_or(terminations, truncations)

        if self.render:
            self.env.render()

    def close(self) -> None:
        self.env.close()
