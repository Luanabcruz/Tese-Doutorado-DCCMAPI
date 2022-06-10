import yaml

def all_var():
    with open(r'./env.yaml') as file:
        env = yaml.load(file, Loader=yaml.FullLoader)
        return env

def datasets_path():
    vars = all_var()
    return vars['datasets_path']

def get_device_id():
    vars = all_var()
    return vars['torch']['device_id']

if __name__ == '__main__':

    print(datasets_path())
    
            