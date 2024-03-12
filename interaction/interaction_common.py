import yaml

with open('interaction/config.yaml', 'r', encoding='UTF-8') as f:
    config = yaml.safe_load(f.read())