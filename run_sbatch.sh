param_pkl=$(srun '--exclude=cn[66-69,71-136,153-256,265-320,325-328,406]' run_container.sh python toy_model_run_nb_with_param_0.py)

for x in ${param_pkl[@]}; do
    sbatch run_container.sh python toy_model_run_nb_with_param_1.py ${x}
done
