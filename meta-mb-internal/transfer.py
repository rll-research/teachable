import argparse
import pathlib
from shutil import copyfile

# Transfer progress.csv and all heldout video folders!


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--folders', nargs='+', required=True, type=str)
parser.add_argument('--copy_model', action='store_true')
args = parser.parse_args()

data_dirs = [
    pathlib.Path('data'),
    pathlib.Path('old_data'),
    pathlib.Path('.'),
]

transfer_folder = pathlib.Path('filetransfer')
if not transfer_folder.exists():
    transfer_folder.mkdir()

def load_run(folder):
    for data_dir in data_dirs:
        d = pathlib.Path(data_dir)
        for run in d.iterdir():
            if folder in run.name:
                return run
    return None

for folder in args.folders:
    data_dir = load_run(folder)
    new_data_dir = transfer_folder.joinpath(folder)
    if not new_data_dir.exists():
        print("making file", new_data_dir.name)
        new_data_dir.mkdir()
    # Transfer everything but the videos and the buffer
    for file_name in data_dir.iterdir():
        if file_name.suffix == '.pkl' and not args.copy_model:
            continue
        if file_name.stem == 'buffer':
            continue
        if file_name.suffix == '.mp4':
            continue
        if file_name.name == 'tb':
            file_name = list(file_name.iterdir())[0]  # events file
        if file_name.is_dir():  # copy recursively
            print("copying folder", file_name.name)
            new_folder = new_data_dir.joinpath(file_name.name)
            if not new_folder.exists():
                new_folder.mkdir()
            for actual_file in file_name.iterdir():
                new_file = new_folder.joinpath(actual_file.name)
                copyfile(actual_file, new_file)
        else:
            print("copying", file_name.stem)
            new_tb_file = new_data_dir.joinpath(file_name.name)
            copyfile(file_name, new_tb_file)


print('scp -r filetransfer/* olivia@pabrtxs1.ist.berkeley.edu:~/Teachable/babyai/meta-mb-internal/data/')
