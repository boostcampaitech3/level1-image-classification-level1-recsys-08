import json


def set_config_as_string(config_namespace, config, section, name):
    if config.has_option(section, name):
        setattr(config_namespace, name, config.get(section, name))


def set_config_as_int(config_namespace, config, section, name):
    if config.has_option(section, name):
        setattr(config_namespace, name, config.getint(section, name))


def set_config_as_float(config_namespace, config, section, name):
    if config.has_option(section, name):
        setattr(config_namespace, name, config.getfloat(section, name))


def set_config_as_json(config_namespace, config, section, name):
    if config.has_option(section, name):
        setattr(config_namespace, name, json.loads(config.get(section, name)))


def set_config_as_bool(config_namespace, config, section, name):
    if config.has_option(section, name):
        setattr(config_namespace, name, config.getboolean(section, name))