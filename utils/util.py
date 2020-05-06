from utils.luatables import LuaEmulator


def convert(yaml_dict):
    """Converts a dictionary to a LuaTable-like object."""
    if isinstance(yaml_dict, dict):
        yaml_dict = LuaEmulator(yaml_dict)
    for key, item in yaml_dict.items():
        if isinstance(item, dict):
            yaml_dict[key] = convert(item)
    return yaml_dict


def merge(config: dict,
          flags):
    for key in flags:
        config[key] = flags[key].value
    return convert(config)

