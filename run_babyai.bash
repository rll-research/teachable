# advices = ("Cardinal", "Waypoint", "OffsetWaypoint", "Direction", "none") SubgoalSimple  PreActionAdvice XYCorrections none

for advice in none
do
    echo baby_$advice
    nohup python scripts/train_model.py --prefix babybase_new_$advice --collect_with_rl_policy --train_level --frames_per_proc 800 --num_envs 2 --rl_teacher $advice --env babyai --device "cuda:0" > run_logs/babybase_new_$advice.log &
done