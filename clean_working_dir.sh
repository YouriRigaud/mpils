#!/bin/bash

# Delete all files in tuner_working_dir and its subdirectories, but keep subdirectories
WORKING_DIR="./tuner_working_dir"
find "$WORKING_DIR" -type f -delete
echo "Cleaned working directory: $WORKING_DIR"

