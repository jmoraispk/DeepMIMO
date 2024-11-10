
import os
import zipfile

def ext_in_list(s, l):
    return [el for el in l if el.endswith(s)]

def zip_folder(folder_path):
    files_in_folder = os.listdir(folder_path)
    file_full_paths = [os.path.join(folder_path, file) 
                       for file in files_in_folder]
    # Create a zip file
    with zipfile.ZipFile(folder_path + '.zip', 'w') as zipf:
        for file_path in file_full_paths:
            zipf.write(file_path, os.path.basename(file_path))
            