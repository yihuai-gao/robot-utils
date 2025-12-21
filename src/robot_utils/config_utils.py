import hydra
from omegaconf import DictConfig, ListConfig, OmegaConf
from typing import Any, cast

def register_resolvers():
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.register_new_resolver("concat", lambda *args: args[0] + args[1])
    OmegaConf.register_new_resolver("cond", lambda cond, true_val, false_val: true_val if cond else false_val)
    OmegaConf.register_new_resolver("range", lambda start, end, step=1: ListConfig(list(range(start, end, step))))
    OmegaConf.register_new_resolver("in", lambda target, *items: any(item in target for item in items))

def disable_hydra_target(cfg: Any) -> Any:
    if isinstance(cfg, DictConfig):
        return_dict = DictConfig({})
        for key, value in cfg.items():
            if key == "_target_":
                return_dict["__target__"] = value
            else:
                return_dict[key] = disable_hydra_target(value)
        return return_dict
    elif isinstance(cfg, ListConfig):
        return_list = ListConfig([])
        for item in cfg:
            return_list.append(disable_hydra_target(item))
        return return_list
    return cfg


def enable_hydra_target(cfg: Any) -> Any:
    if isinstance(cfg, DictConfig) or isinstance(cfg, dict):
        return_dict = DictConfig({})
        for key, value in cfg.items():
            if key == "__target__":
                return_dict["_target_"] = value
            else:
                return_dict[key] = enable_hydra_target(value)
        return return_dict
    elif isinstance(cfg, ListConfig) or isinstance(cfg, list):
        return_list = ListConfig([])
        for item in cfg:
            return_list.append(enable_hydra_target(item))
        return return_list
    return cfg