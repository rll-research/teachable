# advices = ("Cardinal", "Waypoint", "OffsetWaypoint", "Direction", "none")

for advice in Cardinal Waypoint OffsetWaypoint Direction none
do
    echo ant_$advice
    nohup python scripts/train_model.py  --prefix ant_$advice --collect_with_rl_policy --level 3 --frames_per_proc 200 --num_envs 20 --rl_teacher $advice --env ant > ant_$advice.log &
done