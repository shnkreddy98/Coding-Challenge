import yaml

def read_file(file: str):
    with open(file, 'r') as f:
        if file.endswith(".yaml"):
            return yaml.safe_load(f)
        else:
            return f.read()