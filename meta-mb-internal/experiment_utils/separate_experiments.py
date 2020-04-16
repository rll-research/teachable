import argparse
from experiment_utils.utils import load_exps_data
import os
import shutil

"""
 python /home/ignasi/GitRepos/meta-mb/experiment_utils/save_videos.py data/s3/mbmpo-pieter/ --speedup 4 -n 1 --max_path_length 300 --ignore_done
"""


def valid_experiment(params, algo):
    if algo == 'mb-mpo':
        if params['env']['$class'] == 'meta_mb.envs.mb_envs.ant.AntEnv':
            values = {
                  'meta_batch_size': [20],
                  'rolling_average_persitency': [0.4],
                    'dynamics_learning_rate': [0.0005]
                  }
        else:
            values = {
                      'meta_batch_size': [20],
                      'rolling_average_persitency': [0.4],
                      'dynamics_learning_rate': [0.001]
                  }

    elif algo == 'me-ppo':
        if params['env']['$class'] == 'meta_mb.envs.mb_envs.ant.AntEnv':
            values = {
                'num_rollouts': [10],
                'rolling_average_persitency': [0.4],
                'dynamics_learning_rate': [0.0005]
            }
        else:
            values = {
                'rolling_average_persitency': [0.4],
                'dynamics_learning_rate': [0.0005],
            }
        # values = {
        #     'num_rollouts': [10],
        #     'rolling_average_persitency': [0.9],
        #     'clip_eps': [0.2],
        #     'num_ppo_steps': [5],
        #     'learning_rate': [0.0003]
        # }

    elif algo == 'me-trpo':
        if params['env']['$class'] == 'meta_mb.envs.mb_envs.ant.AntEnv':
            values = {
                'rolling_average_persitency': [0.4],
                'dynamics_learning_rate': [0.001]
            }
        else:
            values = {
                'rolling_average_persitency': [0.4],
                'dynamics_learning_rate': [0.001]
            }
        # #Previous:
        # values = {
        #     'num_rollouts': [10],
        #     'rolling_average_persitency': [0.9],
        #     'clip_eps': [0.2],
        #     'num_ppo_steps': [5],
        #     'learning_rate': [0.0003]
        # }

    elif algo == 'a-me-ppo':
        # # Half-Cheetah:
        if params['env'] == 'HalfCheetah':
            values = {
                'clip_eps': [0.2],
                'rolling_average_persitency': [0.4],
                'dynamics_learning_rate': [0.0005],
            }
        elif params['env'] == 'Ant':
            values = {
                'clip_eps': [0.2],
                'rolling_average_persitency': [0.1],
                'dynamics_learning_rate': [0.0005],
            }
        # # Ant:
        # values = {
        #     'clip_eps': [0.2],
        #     'rolling_average_persitency': [0.1],
        #     'dynamics_learning_rate': [0.0005],
        # }
        elif params['env'] == 'Hopper':
            values = {
                'clip_eps': [0.2],
                'rolling_average_persitency': [0.9],
                'dynamics_learning_rate': [0.0005],
            }

        elif params['env'] == 'Walker2d':
            values = {
                'clip_eps': [0.2],
                'rolling_average_persitency': [0.99],
                'dynamics_learning_rate': [0.0005],
            }

    elif algo == 'a-me-trpo':
        if params['env'] == 'HalfCheetah':
            values = {
                'rolling_average_persitency': [0.4],
                'dynamics_learning_rate': [0.0005],
                'step_size': [0.01]
            }
            # values = {
            #     'rolling_average_persitency': [0.1],
            #     'dynamics_learning_rate': [0.001],
            # }

            # values = {
            #     'rolling_average_persitency': [0.4],
            #     'dynamics_learning_rate': [0.0005],
            # }
        elif params['env'] == 'Ant':
            return False
            values = {
                'rolling_average_persitency': [0.1],
                'dynamics_learning_rate': [0.0005],
                'step_size': [0.02],
            }

        elif params['env'] == 'Walker2d':
            values = {
                'rolling_average_persitency': [0.4],
                'dynamics_learning_rate': [0.0005],
                'step_size': [0.01]
            }
            # values = {
            #     'env': ['Walker2d'],
            #     'step_size': [0.05],
            #     'rolling_average_persitency': [0.99],
            #     'dynamics_learning_rate': [0.001],
            # }

            # values = {
            #     'env': ['Walker2d'],
            #     'step_size': [0.05],
                # 'rolling_average_persitency': [0.04],
                # 'dynamics_learning_rate': [0.0005],
            # }

        elif params['env'] == 'Hopper':
            return False
            values = {
                'env': ['Hopper'],
                'step_size': [0.02],
                'rolling_average_persistency': [0.9],
                'dynamics_learning_rate': [0.0005],
                }
        else:
            raise NotImplementedError

    elif algo == 'a-mb-mpo':
        if params['env'] == 'HalfCheetah':
            values = {
                'step_size': [0.05],
                'rolling_average_persitency': [0.1],
                'dynamics_learning_rate': [0.001],

            }
            # values = {
            #     'fraction_meta_batch_size': [0.05],
            #     'rolling_average_persitency': [0.1],
            #     'dynamics_learning_rate': [0.0005],
            # }
        elif params['env'] == 'Ant':
            values = {
                'step_size': [0.05],
                'rolling_average_persitency': [0.1],
                'dynamics_learning_rate': [0.001],


            }
            # values = {
            #     'fraction_meta_batch_size': [0.05],
            #     'rolling_average_persitency': [0.1],
            #     'dynamics_learning_rate': [0.0005],
            # }

        elif params['env'] == 'Walker2d':
            # values = {
            #     'env': ['Walker2d'],
            #     'fraction_meta_batch_size': [0.05],
            #     'rolling_average_persitency': [0.9],
            #     'dynamics_learning_rate': [0.0005],
            # }
            values = {
                'step_size': [0.05],
                'rolling_average_persitency': [0.1],
                'dynamics_learning_rate': [0.0005],

            }

        elif params['env'] == 'Hopper':
            # values = {
            #     'env': ['Hopper'],
            #     'fraction_meta_batch_size': [0.05],
            #     'rolling_average_persitency': [0.4],
            #     'dynamics_learning_rate': [0.0005],
            # }
            values = {
                'step_size': [0.05],
                'rolling_average_persitency': [0.1],
                'dynamics_learning_rate': [0.001],

            }
        else:
            raise NotImplementedError

    elif algo == 'ppo':
        values = {
             'num_rollouts': [50],
             'clip_eps': [0.2],
             'num_ppo_steps': [5],
             'learning_rate': [0.001]
        }

    elif algo == 'trpo':
        values = {}

    elif algo == 'reg':
        values = {
            'algo': ['me-trpo'],
            'num_models': [5],
            'num_grad_policy_per_step': [2],
            'num_epochs_per_step': [1],
            'repeat_steps': [25],
            'rolling_average_persitency': [0.9],
            'exp_tag': ['regularization']
        }

    elif algo == 'no-reg':
        values = {
            'algo': ['me-trpo'],
            'steps_per_iter': [[30, 30]],
            'num_models': [5],
            'rolling_average_persitency': [0.9],
            'exp_tag': ['no-regularization']
        }

    elif algo == 'expl':
        values = {
            'algo': ['me-trpo'],
            'num_rollouts': [10],
            'grad_steps_per_rollout': [4],
            'rolling_average_persitency': [0.9],
            'exp_tag': ['exploration']
        }

    elif algo == 'no-expl':
        values = {
            'algo': ['me-trpo'],
            'num_rollouts': [10],
            'steps_per_iter': [40],
            'rolling_average_persitency': [0.9],
            'exp_tag': ['no-exploration']
        }

    elif algo == 'time-ablations':
        values = {
            'rolling_average_persitency': [0.4],
            'dynamics_learning_rate': [0.0005],
            'step_size': [0.01]
        }

    elif algo == 'persistency-ablations':
        values = {
            'env': ['HalfCheetah', 'Walker2d'],
            'dynamics_learning_rate': [0.0005],
            'step_size': [0.01]
        }
    elif algo == 'a-mb-mpo-pr2':
        return True

    else:
        raise NotImplementedError

    for k, v in values.items():
        try:
            if params[k] not in v:
                return False
        except:
            import pdb; pdb.set_trace()
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=str)
    parser.add_argument("--store", type=str,
            default=os.path.join(os.path.expanduser('~'), 'new'))
    parser.add_argument('--algo', '-a', type=str)
    args = parser.parse_args()

    experimet_paths = load_exps_data(args.data, gap=0.)
    counter = 0
    for exp_path in experimet_paths:
        json_file = exp_path['json']
        if valid_experiment(json_file, args.algo):
            try:
                env_name = json_file['env']['$class'].split('.')[-2]
            except TypeError:
                env_name = json_file['env']
            dir_name = os.path.join(args.store, json_file['algo'], env_name, env_name + str(counter))
            os.makedirs(dir_name, exist_ok=True)
            shutil.copy2(os.path.join(exp_path['exp_name'], "params.json"), dir_name)
            shutil.copy2(os.path.join(exp_path['exp_name'], "progress.csv"), dir_name)
            counter += 1





