import sys, os
root = "resource/material"

for filename in os.listdir(root):
    mtl_path = os.path.join(root, filename)
    with open(mtl_path, "r") as f:
        lines = f.readlines()


    for i in range(len(lines)):
        if lines[i].startswith('diffuse_scattering_model none'):
            lines[i] = lines[i].replace('none', 'directive_with_backscatter')
            print(mtl_path)
    
    lines[1] = "Material 1\n"
    with open(mtl_path, "w") as f:
        f.writelines(lines)

