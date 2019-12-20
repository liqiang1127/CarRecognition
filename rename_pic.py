import os

for parent, subdirs, filenames in os.walk("./test"):
    for i, filename in enumerate(filenames):
        os.rename(os.path.join(parent, filename), os.path.join(parent, "pic" + str(i+1) + ".jpg"))
