import argparse
from babyai.levels.curriculum import Curriculum
import pathlib
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--curriculum_type", default=1, type=int)
parser.add_argument("--save_path", default='', type=str)
args = parser.parse_args()

base_path = pathlib.Path(args.save_path)
if not base_path.exists():
    base_path.mkdir()

curriculum = Curriculum('on_hot', curriculum_type=args.curriculum_type)
for i, env in enumerate(curriculum.levels_list):
    env.reset()
    img = env.render('rgb_array')
    plt.imshow(img)
    plt.title(env.mission)
    plt.savefig(base_path.joinpath(f'Level_{i}_{env.__class__.__name__[6:]}.png'))
