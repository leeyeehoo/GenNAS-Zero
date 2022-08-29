
# https://stackoverflow.com/questions/1305532/how-to-convert-a-nested-python-dict-to-object
class yaml_parser(object):
    def __init__(self, config):
        for k, v in config.items():
            if isinstance(k, (list, tuple)):
                setattr(self, k, [yaml_parser(x) if isinstance(x, dict) else x for x in v])
            else:
                setattr(self, k, yaml_parser(v) if isinstance(v, dict) else v)
