import json, os

def remove_DS_store(path_list):
    paths = [path for path in path_list if '.DS_Store' not in path]
    return paths


def get_raw_name_and_file_type(name):
    raw_name = name[:name.find('.')]
    file_format = name[name.find('.'):]
    return raw_name,file_format


def is_inside(min_x,min_y,max_x,max_y,cx,cy):
    """ will simply return TRUE if the center values liw within a rectangle define in the form: min_x,min_y,max_x,max_y
    else it will return FALSE"""
    if cx>=min_x and cx<=max_x and cy>=min_y and cy<=max_y:
        return True
    else:
        return False

def save_json(file,file_path):
    with open(file_path,'w') as dump:
        json.dump(file,fp=dump)

def read_json(file_path):
    with open(file_path) as decode:
        data = json.load(decode)
    return data

def ensure_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)