import hydra
from omegaconf import DictConfig, ListConfig, OmegaConf
from typing import Any, cast


def register_resolvers():
    def e(x: Any) -> int:
        if isinstance(x, str):
            return eval(x)
        else:
            return x
    def concat(*args: list[ListConfig]) -> ListConfig:
        # Filter out empty lists and start with a fresh ListConfig to avoid aliasing issues
        result = ListConfig([])
        for arg in args:
            if arg is not None and len(arg) > 0:
                result = ListConfig(list(result) + list(arg))
        return result

    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.register_new_resolver("concat", concat)
    OmegaConf.register_new_resolver("cond", lambda cond, true_val, false_val: true_val if cond else false_val)
    OmegaConf.register_new_resolver("range", lambda start, end, step=1: ListConfig(list(range(e(start), e(end), e(step)))))
    OmegaConf.register_new_resolver("in", lambda target, *items: any(item in target for item in items))
    OmegaConf.register_new_resolver("shift", lambda list_cfg, shift: ListConfig([element + e(shift) for element in e(list_cfg)]))


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