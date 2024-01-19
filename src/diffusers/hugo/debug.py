import h5py

current_path = []
saved_paths = {}
save_tensors = False
log = []

def logappend(message):
    log.append(message)
    print(message)

def log_enter():
    path = '/'.join(current_path)
    
    logappend(f"{path}|enter")

def log_exit():
    path = '/'.join(current_path)
    logappend(f"{path}|exit")

def log_save(fn):
    path = '/'.join(current_path)
    logappend(f"{path}|save:{fn}")

def log_tensor(message, tensor):
    path = '/'.join(current_path)
    logappend(f"{path}|{message}:{tensor.shape}")

def get_path_file():
    global current_path, save_paths
    path = '_'.join(current_path)
    if path not in saved_paths:
        save_paths[path]=-1
    save_paths[path]+=1
    return f"{path}_{save_paths[path]}.hdf5"
 
class Operation(object):
    my_path:str
    def __init__(self, path, tensor):
        self.my_path = path
        current_path.append(path)
        log_enter()
        log_tensor("enter",tensor)
        if save_tensors and tensor is not None:
            fn = get_path_file(current_path)
            f = h5py.File(fn, "w")
            f.create_dataset('data', data=tensor)
            log_save(fn)
        
    def __enter__(self):
        pass

    def __exit__(self, *args):
        global current_path
        assert current_path[-1] == self.my_path
        log_exit()
        current_path = current_path[:-1]


def save():
    with open("debg_log.txt",'w') as file:
        file.write('\n'.join(log))

def log_return(tensor):
    log_tensor("return", tensor)