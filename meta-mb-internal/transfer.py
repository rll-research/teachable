import argparse
import pathlib
from shutil import copyfile

# Transfer progress.csv and all heldout video folders!


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--folders', nargs='+', required=True, type=str)
args = parser.parse_args()

data_dirs = [
    pathlib.Path('meta-mb-internal/data'),
    pathlib.Path('meta-mb-internal/old_data'),
    pathlib.Path('meta-mb-internal'),
]

transfer_folder = pathlib.path('transfer')
if not transfer_folder.exists():
    transfer_folder.mkdir()

def load_run(folder):
    for data_dir in data_dirs:
        d = pathlib.Path(data_dir)
        for run in d.iterdir():
            if folder in run.name:
                return folder
    return None

for folder in args.folders:
    data_dir = load_run(folder)
    new_data_dir = transfer_folder.joinpath(folder.name)
    if not new_data_dir.exists():
        new_data_dir.mkdir()
    # Transfer everything but the videos and the buffer
    for file_name in data_dir.iterdir():
        if file_name == 'buffer':
            continue
        if file_name.suffix == '.mp4':
            continue
        new_tb_file = data_dir.joinpath(file_name.name)
        copyfile(file_name, new_tb_file)


print('scp -r transfer/.* olivia@pabrtxs1.ist.berkeley.edu:~/Teachable/babyai/meta-mb-internal/data/')