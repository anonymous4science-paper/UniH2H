                                             
                      
 
                                                                    
                                                                             
 
                                                                                
                                                     
 
                                                                              
                                                                              
                                                           
 
                                                                  
                                                                         
                                                             
 
                                                                             
                                                                           
                                                                                
                                                                              
                                                                            
                                                                            
                                                                            
                                                                               
                                                                               
                                                                      

        
            
import getpass
import tempfile
import time
from collections import OrderedDict
from os.path import join

import numpy as np
import torch
import random
import os


def retry(times, exceptions):
    """
    Retry Decorator https://stackoverflow.com/a/64030200/1645784
    Retries the wrapped function/method `times` times if the exceptions listed
    in ``exceptions`` are thrown
    :param times: The number of times to repeat the wrapped function/method
    :type times: Int
    :param exceptions: Lists of exceptions that trigger a retry attempt
    :type exceptions: Tuple of Exceptions
    """

    def decorator(func):
        def newfn(*args, **kwargs):
            attempt = 0
            while attempt < times:
                try:
                    return func(*args, **kwargs)
                except exceptions:
                    print(
                        f"Exception thrown when attempting to run {func}, attempt {attempt} out of {times}"
                    )
                    time.sleep(min(2**attempt, 30))
                    attempt += 1

            return func(*args, **kwargs)

        return newfn

    return decorator


def flatten_dict(d, prefix="", separator="."):
    res = dict()
    for key, value in d.items():
        if isinstance(value, (dict, OrderedDict)):
            res.update(flatten_dict(value, prefix + key + separator, separator))
        else:
            res[prefix + key] = value

    return res


def set_np_formatting():
    """formats numpy print"""
    np.set_printoptions(
        edgeitems=30,
        infstr="inf",
        linewidth=4000,
        nanstr="nan",
        precision=2,
        suppress=False,
        threshold=10000,
        formatter=None,
    )


def set_seed(seed, torch_deterministic=False, rank=0):
    """set seed across modules"""
    if seed == -1 and torch_deterministic:
        seed = 42 + rank
    elif seed == -1:
        seed = np.random.randint(0, 10000)
    else:
        seed = seed + rank

    print("Setting seed: {}".format(seed))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch_deterministic:
                                                                                           
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    return seed


def nested_dict_set_attr(d, key, val):
    pre, _, post = key.partition(".")
    if post:
        nested_dict_set_attr(d[pre], post, val)
    else:
        d[key] = val


def nested_dict_get_attr(d, key):
    pre, _, post = key.partition(".")
    if post:
        return nested_dict_get_attr(d[pre], post)
    else:
        return d[key]


def ensure_dir_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def safe_ensure_dir_exists(path):
    """Should be safer in multi-treaded environment."""
    try:
        return ensure_dir_exists(path)
    except FileExistsError:
        return path


def get_username():
    uid = os.getuid()
    try:
        return getpass.getuser()
    except KeyError:
                                                  
        return str(uid)


def project_tmp_dir():
    tmp_dir_name = f"ige_{get_username()}"
    return safe_ensure_dir_exists(join(tempfile.gettempdir(), tmp_dir_name))


     
