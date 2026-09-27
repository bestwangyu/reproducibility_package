import sys
import types


def install_gym_stub_if_needed() -> None:
    try:
        __import__("gym")
        return
    except ModuleNotFoundError:
        pass

    gym_module = types.ModuleType("gym")
    gym_module.Env = object
    envs_module = types.ModuleType("gym.envs")
    registration_module = types.ModuleType("gym.envs.registration")
    registration_module.register = lambda **kwargs: None
    envs_module.registration = registration_module
    gym_module.envs = envs_module

    sys.modules["gym"] = gym_module
    sys.modules["gym.envs"] = envs_module
    sys.modules["gym.envs.registration"] = registration_module
