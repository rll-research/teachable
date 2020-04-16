# bash script to search through the hyperparams

env_name=$1

for svg_learning_rate in 0.0001 0.0003 0.001; do
    for kl_penalty in 0.001 0.003 0.01; do
        for seed in 1234 2345 3456 4567; do
            exp_name_prefix=kl_${kl_penalty}_lr_${svg_learning_rate}_
            echo "Running ${exp_name_prefix}${env_name}_seed_${seed}"
            python run_scripts_bm/svg_run_tingwu.py \
                --env_name $env_name \
                --exp_name_prefix $exp_name_prefix \
                --seed $seed \
                --kl_penalty ${kl_penalty} \
                --svg_learning_rate ${svg_learning_rate}
        done
    done
done
