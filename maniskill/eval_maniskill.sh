BASE_PATH=../results_approximator/maniskill_weightgen

parallel -ut -j16 python ppo_eval.py \
    --env_id {3}-v0 \
    --length {1} \
    --ckpt_path ${BASE_PATH}/{2}/length/{3}/seed_2/models/input_param{1}.pth \
    --exp_name {2}-length ::: $(seq -f "%.3f" 0.5 0.1 2.0) ::: hypogen ::: LiftCube PickCube

parallel -ut -j16 python ppo_eval.py \
    --env_id {3}-v0 \
    --cube_size {1} \
    --ckpt_path ${BASE_PATH}/{2}/cube/{3}/seed_2/models/input_param{1}.pth \
    --exp_name {2}-cube ::: $(seq -f "%.3f" 0.01 0.002 0.03) ::: hypogen ::: LiftCube PickCube

parallel -ut -j16 python ppo_eval.py \
    --env_id {3}-v0 \
    --agent_stiffness {1} \
    --ckpt_path ${BASE_PATH}/{2}/stiff/{3}/seed_2/models/input_param{1}.pth \
    --exp_name {2}-stiff ::: $(seq -f "%.3f" 500 100 1500) ::: hypogen ::: LiftCube PickCube

parallel -ut -j16 python ppo_eval.py \
    --env_id {3}-v0 \
    --agent_damping {1} \
    --ckpt_path ${BASE_PATH}/{2}/damp/{3}/seed_2/models/input_param{1}.pth \
    --exp_name {2}-damp ::: $(seq -f "%.3f" 50 10 150) ::: hypogen ::: LiftCube PickCube
