#!/bin/bash

VENV_NAME="venv"

python3 -m venv $VENV_NAME

source $VENV_NAME/bin/activate

if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "No requirements.txt found. Skipping installation."
fi

echo "Virtual environment setup complete!"
