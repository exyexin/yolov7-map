import os
def get_gt(path,filename):
    name=os.path.join(path,filename)
    with open(name,'r') as file:
        for line in file:
            print(line)
        return line.split()      