from utils.luatables import LuaEmulator


def convert(yaml_dict):
    """Converts a dictionary to a LuaTable-like object."""
    if isinstance(yaml_dict, dict):
        yaml_dict = LuaEmulator(yaml_dict)
    for key, item in yaml_dict.items():
        if isinstance(item, dict):
            yaml_dict[key] = convert(item)
    return yaml_dict


# def deep_merge_dict(dict_x, dict_y, path=None):
#     """Recursively merges dict_y into dict_x."""
#     if path == None: path = []
#     for key in dict_y:
#         if key in dict_x:
#             if isinstance(dict_x[key], dict) and isinstance(dict_y[key], dict):
#                 deep_merge_dict(dict_x[key], dict_y[key], path + [str(key)])
#             elif dict_x[key] == dict_y[key]:
#                 pass  # same leaf value
#             else:
#                 dict_x[key] = dict_y[key]
#         else:
#             dict_x[key] = dict_y[key]
#     return dict_x


def merge(config: dict,
          flags):
    for key in flags:
        config[key] = flags[key].value
    return convert(config)
