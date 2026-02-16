#!/bin/bash

# Delete all files in tuner_working_dir and its subdirectories, but keep subdirectories
# Clean also the subdirectories ./tuner_working_dir/param_ils/paramils-out*
WORKING_DIR="./tuner_working_dir"
find "$WORKING_DIR" -type f -delete
find "$WORKING_DIR/param_ils" -maxdepth 1 -type d -name "paramils-out*" -exec rm -rf {} +
echo "Cleaned working directory: $WORKING_DIR"

# Clean clone*.log cplex files and cplex.log
find . -type f \( -name "clone*.log" -o -name "cplex.log" \) -delete
echo "Cleaned clone*.log and cplex.log files"