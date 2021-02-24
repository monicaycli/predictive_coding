param_pkl=$(python toy_model_run_nb_with_param_0.py)

for x in ${param_pkl[@]}; do
    python toy_model_run_nb_with_param_1.py ${x}
done