"""
This file contains some miscellaneous helper functions.
"""

from os import mkdir, listdir
from os.path import isdir as folder_exists
from os.path import isfile, splitext, basename

def file_exists(path):
    """
    Check if a file exists
    
    Inputs
    ----------
    path: str
        Path to file.
       
    Outputs
    -------
    bool
        True if file exists, false otherwise.
        
    """

    if path is None:
        return False

    if not isfile(path):
        return False

    return True

def folder_guard(path):
    if not folder_exists(path):
        print('INFO:folder_guard(): Creating folder: ' + path + '...')
        mkdir(path)

def folder_is_empty(path):
    """
    Check if a folder is empty. If the folder does not exist, it counts as being empty. 
    
    Inputs
    ----------
    path: str
        Path to folder.
       
    Outputs
    -------
    bool
        True if folder exists and contains elements, false otherwise.
        
    """

    if folder_exists(path):
        return (len(listdir(path)) == 0)
    
    return True
