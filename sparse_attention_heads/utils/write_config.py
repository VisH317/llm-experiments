from configparser import ConfigParser

def write_config(path: str, d: dict):
    config = ConfigParser()
    config["train"] = d["train"]
    config["transformer"] = d["transformer"]
    config["vocab"] = d["vocab"]

    with open(path, "w+") as f:
        config.write(f)