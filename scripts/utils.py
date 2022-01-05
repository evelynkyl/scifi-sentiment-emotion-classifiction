import os

def read_file(file_path):
    """ 
    Read all the files in a directory to a nested list
    @param    text (str): path to the directory
    @return   text(list of lists): nested list of documents
    """
    files = os.listdir(file_path)
    txt_list = []
  #  keyword = '.txt'
    for file in files:
        if os.path.isfile(os.path.join(file_path, file)):
            with open(os.path.join(file_path, file),'r') as f:
                data = f.read().splitlines()
                txt_list.append(data)
    return txt_list
  
  
def read_file_str(file_path):
    """ 
    Read all the files in a directory to a list of strings
    @param    text (str): path to the directory
    @return   text(list of strings): list of documents as string, each document is a string
    """
    files_list = []
    files = os.listdir(file_path)
    for file in files:
      if os.path.isfile(os.path.join(file_path, file)):
        with open(os.path.join(file_path, file),'r') as f:
          data = f.read() #rstrip()
          data = data.replace("\n", " ")
          files_list.append(data)
    return files_list
  
  
def get_names(file_path):
    """ 
    Function to get the name of the file
    @param    text (str): path to the directory
    @return   text(list of strings): list of document names
    """
    files = os.listdir(file_path)
    list_of_names = []
    for file in files:
      filename = (file.split("__", 1)[1])
      filename = filename[1:-4]
      list_of_names.append(filename)
    return list_of_names
      

def remove_dot_line(inlist):
    """ 
    Function to remove line that contains only a dot i.e., "."
    @param    text (list of strings): lit of documents 
    @return   text(list of strings): list of cleaned documents
    """
    newlist = []
    for book in inlist:
        new_book = []
        for sent in book:
            if sent != "." or sent !=". ":
               new_book.append(sent)
            else:
                pass
        newlist.append(new_book)
    return newlist


def flatten_list(nested_list):
    """ 
    Function to flatten a nested list
    @param    text (nested list)
    @return   text(list)
    """
    return [item for items in nested_list for item in items]