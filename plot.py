from cgi import print_form
from unittest import main
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import pathlib
from matplotlib.pyplot import cm
import matplotlib.lines as mlines
plt.rcParams['figure.figsize'] = [15, 5]
import seaborn as sbn
import matplotlib as mpl

data_dirs = [
    pathlib.Path('logs'),
#     pathlib.Path('meta-mb-internal/eval_runs'),
#     pathlib.Path('meta-mb-internal/old_data'),
#     pathlib.Path('meta-mb-internal/old_eval_runs'),
#     pathlib.Path('meta-mb-internal'),
]

def get_filenames(name):
    name = str(name)
    name_paths = []
    for data_dir in data_dirs:
        for exp_dir in data_dir.iterdir():
            if name in exp_dir.name:
                name_paths.append(exp_dir)
    print("NP", name_paths)
    return name_paths

def format_plot():
    fig, ax = plt.subplots(1, 1)
    sbn.set_style('darkgrid')
    ax.get_xaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax.grid(b=True, which='major', color='w', linewidth=1.0)
    ax.grid(b=True, which='minor', color='w', linewidth=0.5)

def plot_curriculum(data_names, x_type, use_min_len=True, y='Curriculum Percent', smooth=1,
                   level_start=0, level_end=float('inf'), title=None, error_bars=False, y_lower=0,
                   y_upper=1.01, plot_advancement=False, y_label="default", y_transform=lambda x: x, 
                    save_name=None, x_min=None, x_max=None):
    
    if not type(y) is list:
        y = [y]
    format_plot()
        
    color = iter(cm.rainbow(np.linspace(0, 1, len(data_names) * len(y))))
    plots = []
        
    for y_str in y:
        df_y_key = y_str
        df_level_key = 'Level'
        if x_type == 'samples':
            x_label = 'Samples'
            df_x_key = 'Itr'
            title = 'Sample Efficiency' if title is None else title
        elif x_type == 'feedback':
            x_label = 'Advice Units'
            df_x_key = 'Train/NumFeedbackTotal'
            title = 'Feedback Efficiency' if title is None else title
            
        print("data names", data_names)

        for run_names, legend_name, c, marker in data_names:
            print("RUN NAMES", run_names)
            if len(y) > 1:
                legend_name = legend_name + ' - ' + y_str
            try:
                if c is None:
                    c = next(color)
                print("about to load")
                data = [d for run_name in run_names for d in load_data(run_name)]
                print("got data", len(data))
                cols = [df[df_y_key].ewm(span=smooth).mean().to_numpy() for df in data]
                print(cols[-5:])
                print("LENGTHS", [len(d) for d in data])
                shared_good_indices = np.arange(len(data[0]))
                for i, run in enumerate(cols):
                    print("go to cols")
                    run = y_transform(run)
                    x_val = data[i][df_x_key].to_numpy()
                    non_null_indices = ~np.isnan(x_val)
                    levels = data[i][df_level_key].to_numpy()
                    above_min_level = levels >= level_start
                    below_max_level = levels <= level_end
                    in_range = np.logical_and(above_min_level, below_max_level)
                    good_indices = np.logical_and(in_range, non_null_indices)
                    good_indices = np.argwhere(good_indices)[:, 0]
                    if error_bars:
                        print("error_bars")
                        shared_good_indices = [x for x in shared_good_indices if x in good_indices]
                    else:
                        print(f"x_val is {x_val}")
                        plt.plot(x_val[good_indices], run[good_indices], c=c)
                if plot_advancement:
                    level_advancement = data[i][df_level_key].diff().to_numpy()
                    advancement_indices = np.argwhere(level_advancement > 0)[:, 0]
                    for advancement_index in advancement_indices:
                        plt.axvline(x=advancement_index, c=c)
                if error_bars:
                    run_data = np.stack(y_transform(run[shared_good_indices]) for run in cols)
                    mean_run_data = np.mean(run_data, axis=0)
                    std_run_data = np.std(run_data, axis=0)
                    x_val = data[0][df_x_key].to_numpy()[shared_good_indices]

                    shortest_len = x_val[-1]
#                     if x_max is not None:
#                         shortest_len = x_max
                    x = np.arange(0, shortest_len, max(shortest_len // 50, 1))
                    mean_run_data = np.interp(x, x_val, mean_run_data)
                    std_run_data = np.interp(x, x_val, std_run_data)
                    x_val = x
                    plt.plot(x_val, mean_run_data, c=c, marker=marker)
                    plt.fill_between(x_val, mean_run_data - std_run_data, mean_run_data + std_run_data, color=c, alpha=0.2)
                legend_line = mlines.Line2D([], [], color=c, marker=marker, label=legend_name)
                plots.append(legend_line)
            except Exception as e:
                print(e)
#                 print("couldn't plot " + legend_name)
# peacock
    plt.ticklabel_format(scilimits=(0,0))
    plt.title(title, fontsize=20)
    if y_label == 'default':
        y_label = y_str if len(y) else ''
    plt.ylabel(y_label, fontsize=20)
    plt.ylim(y_lower, y_upper)
    plt.xlabel(x_label, fontsize=20)
#     plt.legend(handles=plots, fontsize=20)
    plt.margins(x=0)
    if x_max is not None:
        if x_min is None:
            x_min = 0
        plt.xlim(x_min, x_max)
    if save_name is not None:
        plt.savefig('figs/' + save_name, bbox_inches='tight')
    plt.show()


def load_data(name, file_name='progress.csv'):
    if type(name) is pathlib.PosixPath:
        name_paths = [name]
    else:
        name_paths = get_filenames(name)
    assert len(name_paths) > 0, "No files found with name " + name
    # check that they're all the same except the seed
    print("nps", len(name_paths))
    csv_datas = []
    for path in name_paths:
#         assert path.name[:-1] == name_paths[0].name[:-1], f"Found multiple files with name {name}: {[path.name for path in name_paths]}"
        try:
            csv_name = path.joinpath(file_name)
            data = pd.read_csv(csv_name)
        except Exception as e:
            print(e)
            csv_name = path.joinpath('new_progress.csv')
            data = pd.read_csv(csv_name)
            
        # Remove whitespace in col names
        data.columns = [c.strip() for c in data.columns]
        csv_datas.append(data)
    print("LL", len(csv_datas))
    return csv_datas
        
def plot_final_generalization(runs, title, metric='success_rate', all_run_names=None, all_env_names=None,
                             save_name=None):
    env_dict = {}
    for run_names, legend_name in runs:
        try:
            data = []
            for run_name in run_names:
                data += load_data(run_name, file_name='results.csv')
            success = [df[metric].to_numpy() for df in data]
            env_names = data[0]['env'].to_numpy()
            for i, env_name, in enumerate(env_names):
                if all_run_names is not None and not env_name in all_run_names:
                    continue
                if not env_name in env_dict:
                    env_dict[env_name] = ([], [], [])
                mean_list, std_list, legend_list = env_dict[env_name]
                mean_list.append(np.mean([s[i].item() for s in success]))
                std_list.append(np.std([s[i].item() for s in success]))
                legend_list.append(legend_name)
        except Exception as e:
            print("Error on ", run_names, e)
    print("ENV DICT", env_dict.keys())
    colors = ['blue', 'green', 'orange', 'purple', 'red', 'yellow', 'brown', 'pink']
    for i, (env_name, (mean_list, std_list, legend_list)) in enumerate(env_dict.items()):
        if all_env_names is not None:
            env_title = all_env_names[i]
        else:
            env_title = title
        plt.title(env_title, fontsize=20)
#         plt.title(title + ' - ' + str(env_name) + '; ' + metric, fontsize=20)
        color = 'green' if 'accuracy' in metric else None
        plt.bar(legend_list, mean_list, yerr=std_list, capsize=5, color=color)
        plt.xticks(rotation=80, fontsize=20)
        plt.ylim(0, 1)
        if save_name is not None:
            plt.savefig('figs/' + save_name + env_title + '.png', bbox_inches='tight')
        plt.show()
        

def plot_curriculum_generalization(runs, title, metric='success_rate'):
    for run_names, legend_name in runs:  # Currently plotting each one separately.
        try:
            data = []
            for run_name in run_names:
                data += load_data(run_name, file_name='results.csv')
            levels = sorted(np.unique(data[0]['policy']))
            print("LEVELS", levels)
            for level in levels:
                relevant_data = [df[df['policy'] == level] for df in data]
                x = relevant_data[0]['env']
                y_data = [df[metric].to_numpy() for df in relevant_data]
                y_mean = y_data[0]
                y_err = np.zeros_like(y_mean)
                y_mean = [np.mean([y[i] for y in y_data]) for i in range(len(y_data[0]))]
                y_err = [np.std([y[i] for y in y_data]) for i in range(len(y_data[0]))]
                plt.title(title + ' - ' + str(level))
                color = 'green' if 'accuracy' in metric else None
                plt.bar(x, y_mean, yerr=y_err, capsize=5, color=color)
                plt.xticks(rotation=90)
                plt.show()
        except Exception as e:
            print("Error on ", run_names, e)

        
def plot_leaderboard(methods):
    plot_values = []
    error_bars = []
    plot_names = []
    num_feedback_key = 'Train/NumFeedbackTotal'
    for run_names, legend_name, _ in methods:
        try:
            data = []
            for run_name in run_names:
                data += load_data(run_name)
            if run_name in ['0577', '0578']:  # TODO: delete this later now that we've fixed how we count feedback
                pure_feedback_list = [df['Train/NumFeedbackAdvice'].to_numpy()[-2].item() for df in data]
                reward_list = [df['Train/NumFeedbackReward'].to_numpy()[-2].item() for df in data]
                path_length_list = [df['Train/PathLength'].to_numpy()[-2].item() for df in data]
                success_list = [df['Train/Success'].to_numpy()[-2].item() for df in data]
                num_feedback_list = [pure_feedback + reward_feedback / path_length * success_rate for reward_feedback, path_length, success_rate, pure_feedback in zip(reward_list, path_length_list, success_list, pure_feedback_list)]
            else:
                num_feedback_list = [df[num_feedback_key].to_numpy()[-2].item() for df in data]
            plot_values.append(np.mean(num_feedback_list))
            error_bars.append(np.std(num_feedback_list))
            plot_names.append(f'{run_name} - {legend_name}')
        except Exception as e:
            temp = 3
#             print(e)
#             print("couldn't plot " + legend_name)
    plt.bar(plot_names, plot_values, yerr=error_bars, capsize=5)
    plt.xticks(rotation=45)
    plt.show()

# NOTE: this is just here temporarily to make the csvs parsable b/c of a bug.
# The bug is fixed now, so once the current runs become unnecessary (or have all been fixed) we can delete this
def fix_csv(filename):
    data_folder = get_filenames(filename)[0]
    correct_data = []
    csv = data_folder.joinpath('progress.csv')
    new_csv = data_folder.joinpath('new_progress.csv')
    print(csv)
    with open(csv, 'r') as f:
        for line in f.readlines():
            if 'BufferSize' in line:
                correct_data = []
            correct_data.append(line)
    with open(new_csv, 'w') as f:
        f.writelines(correct_data)

def plot_finetune_generalization(runs, title, metric='success_rate', smooth=1, 
                                 x_label='Itrs',
                                 all_run_names=None, y_label='default', 
                                 y_transform=lambda x: x, y_min=0, y_max=1.01, save_name=None,
                                 x_min=None, x_max=None):
    env_names = None
    env_dict = {}
    for run_info in runs:
        c = None
        marker = None
        if len(run_info) == 5:
            run_names, legend_name, feedback_per_timestep, c, marker = run_info
            use_itrs = x_label in ['Itrs', 'Samples']
        elif len(run_info) == 4:
            run_names, legend_name, feedback_per_timestep, use_itrs = run_info
        elif len(run_info) == 3:
            run_names, legend_name, feedback_per_timestep = run_info
            use_itrs = x_label in ['Itrs', 'Samples']
        else:
            run_names, legend_name = run_info
            feedback_per_timestep = 1
            use_itrs = x_label  in ['Itrs', 'Samples']
        try:
            data = []
            env_data = {}
            for run_name in run_names:
                name_paths = get_filenames(run_name)
                assert len(name_paths) == 1, ("Didn't find exactly 1 run name", name_paths)
                name_path = name_paths[0]
                for sub_path in name_path.iterdir():
#                     print("SUB PATH", sub_path)
                    if not 'checkpoint' in sub_path.name:
                        continue
                    data = load_data(sub_path, file_name='results.csv')[0]
                    env_name = data.iloc[0]['env']
                    success = data[metric].ewm(span=smooth).mean().to_numpy() 
                    if all_run_names is not None and not env_name in all_run_names:
                        continue
                    if not env_name in env_data:
                        env_data[env_name] = ([], [])
                    feedback_list, data_list = env_data[env_name]
                    data_list.append(success)
                    # TODO: delete last case
                    if use_itrs or not 'num_feedback' in data.columns or legend_name == 'Distill from Oracle':
                        scale_factor = 1 if x_label == 'Itrs' else feedback_per_timestep
                        feedback_list.append(np.arange(len(success)).astype(np.float32) * scale_factor)
                    elif 'num_feedback' in data.columns:
                        x = data['num_feedback'].to_numpy() * feedback_per_timestep
                        
                        # TODO: remove this later!
                        if run_name in [ 'T3654', 'T3655','T3656', 'T3657', 'T3658', 'T3659', 'T3660', 'T3765', 'T3766', 'T3767', 'T3768'
]:
                            x = x - x[0]
#                         print("X", x[:5] / data['itr'].to_numpy()[:5] / 800)
                        feedback_list.append(x)
#                     print("X", feedback_list[-1][:20])
            for env_name, (env_x_list, env_y_list) in env_data.items():
                if not env_name in env_dict:
                        env_dict[env_name] = ([], [], [], [], [], [])
                policy_list, x_list, y_list, y_std_list, color_list, marker_list = env_dict[env_name]
                policy_list.append(legend_name)
                print("LENGTHS", [x[-1] for x in env_x_list])
                shortest_len = min([x[-1] for x in env_x_list])
#                 if x_max is not None:
#                     shortest_len = x_max
#                     print("setting shortest len to X_max", x_max)
                x = np.arange(0, shortest_len, max(shortest_len // 50, 1))
                run_data = np.stack([np.interp(x, run_x, run_y) for run_x, run_y in zip(env_x_list, env_y_list)])
                mean_run_data = np.mean(run_data, axis=0)
                std_run_data = np.std(run_data, axis=0)
                y_list.append(mean_run_data)
                y_std_list.append(std_run_data)
                x_list.append(x)
                color_list.append(c)
                marker_list.append(marker)
        except Exception as e:
            print("Error on ", run_names, e)
    for env_name, (policy_list, x_list, y_list, y_std_list, color_list, marker_list) in env_dict.items():
        plt.ticklabel_format(scilimits=(0,0))
        plt.title(title, fontsize=15)
        if x_max is not None:
            if x_min is None:
                x_min = 0
            plt.xlim(x_min, x_max)
#         plt.title(title + ' - ' + str(env_name), fontsize=20)
        for x_data, y_data, y_std_data, color, marker in zip(x_list, y_list, y_std_list, color_list, marker_list):
            print("plotting", color)
            plt.plot(x_data, y_transform(y_data), c=color, marker=marker)
            plt.fill_between(x_data, y_data - y_std_data, y_data + y_std_data, alpha=0.2, color=color)
#         plt.legend(policy_list, fontsize=20)
        y_label = metric if y_label == 'default' else y_label
        plt.ylabel(y_label, fontsize=15)
        plt.xlabel(x_label, fontsize=15)
        if metric in ['success_rate', 'stoch_accuracy']:
            plt.ylim(y_min, y_max)
        if save_name is not None:
            plt.savefig('figs/' + save_name, bbox_inches='tight')
        plt.show() # peacock
        
# #         plt.box(False)
#         for x_data, y_data, y_std_data in zip(x_list, y_list, y_std_list):
# #             print("MAX", np.max(y_transform(y_data)))
#             plt.plot(x_data, y_transform(y_data))
#             plt.fill_between(x_data, y_data - y_std_data, y_data + y_std_data, alpha=0.2)
#             plt.xlim(-10, 0)
#             plt.ylim(-2, -1)
#             plt.legend(policy_list, fontsize=20)
#         plt.box(False)
#         plt.axis('off')
#         if save_name is not None:
#             plt.savefig('figs/legend_' + save_name)
#         plt.show()
        
def add_line(height, text, color=None):
    plt.axhline(height, color=color)
    plt.text(0, height + .01, text)
    
    
if __name__ == "__main__":
    metrics = [
    'Train/Success',
    ]


    data_names = [
        (['EXP_NAME'], 'OffsetWaypoint', 'blue', ''),
    ]


    plot_curriculum(data_names, 'feedback', 
                y=metrics, 
                smooth=50, error_bars=False, 
                title="Train with RL, Dense Reward",
                y_lower=0, y_upper=1, y_label="Success", save_name="pmo1")

