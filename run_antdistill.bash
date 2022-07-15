# advices = ("Cardinal", "Waypoint", "OffsetWaypoint", "Direction", "none")

for advice in Cardinal Waypoint OffsetWaypoint Direction none
do
    echo ant_$advice
    nohup python scripts/train_model.py --min_itr_steps 50000000 --distill_teacher none --rl_policy logs/ant_$advice  --prefix antdistill_$advice --collect_with_rl_policy --level 3 --frames_per_proc 200 --num_envs 20 --rl_teacher $advice --env ant > antdistill_$advice.log &
done