# advices = ("Cardinal", "Waypoint", "OffsetWaypoint", "Direction", "none")

for advice in Cardinal Waypoint OffsetWaypoint Direction none
do
    echo pm_$advice
    nohup python scripts/train_model.py  --prefix pmbase_$advice --train_level --collect_with_rl_policy --level 4 --frames_per_proc 200 --num_envs 20 --rl_teacher $advice --env point_mass > pmbase_$advice.log &
done