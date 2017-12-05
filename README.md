# a2c-acktr-vizdoom
A2C and ACKTR implementations for ViZDoom

# Packages to install
pytorch
scipy
sdl2
vizdoom


## Training the agent
~~~~
python main.py --algo a2c --num-processes 16 --config-path scenario/health_gathering.cfg --num-frames 10000000 --log-dir $LOG_DIR --save-dir $SAVE_DIR --no-vis
~~~~

## Running the agent

MODEL_FILE_NAME: name of the model without the .pt extenstion (assumes the model is in ./trained_models)
~~~~
python enjoy.py --config-path scenario/health_gathering.cfg --env-name MODEL_FILE_NAME
~~~~

