import os

home_dir = os.path.expanduser('~')
username = os.path.basename(home_dir)
gdata_dir = os.path.join("/g/data/xe2", username)
scratch_dir = os.path.join('/scratch/xe2', username)
paddockTS_dir = os.path.join(home_dir, "Projects/PaddockTS")

if os.path.expanduser("~").startswith("/home/"):
    paddockTS_dir = os.path.join(os.path.expanduser("~"), "Projects/PaddockTS")
else:
    paddockTS_dir = os.path.dirname(os.getcwd())


if __name__ == '__main__':
    print("username:", username)
    print("home_dir:", home_dir)
    print("gdata_dir:", gdata_dir)
    print("scratch_dir:", scratch_dir)
