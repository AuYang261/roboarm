# Description: 负责配置参数的读取，以免要在各个模块中重复编写相同的load代码

import os
import yaml
from typing import Any


def load_config(config_file: str) -> dict:
    """
    加载 YAML 配置文件
    """
    if not os.path.exists(config_file):
        raise FileNotFoundError(
            f"配置文件未找到: {config_file}。请按 {config_file}.example 创建配置文件{config_file}"
        )

    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


CONFIG = load_config(os.path.join(os.path.dirname(__file__), "config.yaml"))


def get_config_value(key: str, default=None, raise_if_missing: bool = True) -> Any:
    """
    获取配置文件中的值
    """
    if raise_if_missing and key not in CONFIG:
        raise KeyError(f"配置项未找到: {key}")
    return CONFIG.get(key, default)
