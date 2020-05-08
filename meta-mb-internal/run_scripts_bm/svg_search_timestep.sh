# bash script to search through the hyperparams

env_name=$1
num_iteration=$2  # 50 for 200k timesteps, 250 for 1m, 2500 for 10m

for svg_learning_rate in 0.0001 0.0003; do
    for kl_penalty in 0.001 0.003; do
        for seed in 1234 2345 3456 4567; do
            exp_name_prefix=kl_${kl_penalty}_lr_${svg_learning_rate}_numiteration_${num_iteration}
            echo "Running ${exp_name_prefix}${env_name}_seed_${seed}_iteration_${num_iteration}"
            python run_scripts_bm/svg_run_tingwu.py \
                --env_name $env_name \
                --exp_name_prefix $exp_name_prefix \
                --seed $seed \
                --kl_penalty ${kl_penalty} \
                --svg_learning_rate ${svg_learning_rate} \
                --num_iteration ${num_iteration}
        done
    done
done
