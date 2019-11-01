DATA_PATH="inputs/bluebook-for-bulldozers/TrainAndValid.csv"
PARAMS_PATH="training/params.yml"
MODEL_NAME="RandomForestRegressor"

python -m training.experiment \
    --data_path $DATA_PATH \
    --params_path $PARAMS_PATH \
    --model_name $MODEL_NAME \
    "$@"