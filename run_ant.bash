# advices = ("Cardinal", "Waypoint", "OffsetWaypoint", "Direction", "none")

for advice in Cardinal Waypoint OffsetWaypoint Direction none
do
    echo ant_$advice
    nohup python scripts/train_model.py  --prefix antbase_$advice --collect_with_rl_policy --level 6 --train_level --frames_per_proc 200 --num_envs 20 --rl_teacher $advice --env ant --device "cuda:1" > antbase_$advice.log &
done