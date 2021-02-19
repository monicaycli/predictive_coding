ipynb_list=(\
toy_model_kalman_variant_4.ipynb \
toy_model_kalman_variant_3a.ipynb \
toy_model_kalman_variant_2a.ipynb \
toy_model_kalman_variant_1.ipynb \
)

param_pkl=$(python toy_model_run_nb_with_param_0.py)

for x in ${ipynb_list[@]}; do
    for y in ${param_pkl[@]}; do
        python toy_model_run_nb_with_param_1.py ${x} ${y}
    done
done