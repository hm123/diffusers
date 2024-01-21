import h5py
from inspect import getframeinfo, stack
    

current_path = []
saved_paths = {}
save_tensors = False
log = []
remembered_tensors = {}

def logappend(message):
    log.append(message)
    print(message)

def log_enter(caller):
    path = '/'.join(current_path)
    
    logappend(f"{path}|enter={caller}")

def log_exit():
    path = '/'.join(current_path)
    logappend(f"{path}|exit")

def log_save(fn):
    path = '/'.join(current_path)
    logappend(f"{path}|save:{fn}")

callback = None
callback_attention = None
def log_attention(query, key, value, output):
    path = '/'.join(current_path)
    if callback_attention is not None:
        callback_attention(path, query, key, value, output)

    
def log_tensor(message, tensor, remember = False):
    path = '/'.join(current_path)
    logappend(f"{path}|{message}:{tensor.shape if tensor is not None else 'None'}")
    if remember:
        remembered_tensors[message] = tensor
    if callback is not None:
        callback(current_path, message, tensor)    

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
        caller = getframeinfo(stack()[1][0])
        caller = f"{caller.filename}:{caller.lineno}"
        self.my_path = path
        current_path.append(path)
        log_enter(caller)
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