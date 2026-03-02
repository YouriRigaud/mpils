#!/bin/bash

# Create the working directory if it doesn't exist with all its subdirectories
WORKING_DIR="./tuner_working_dir"
mkdir -p "$WORKING_DIR"
echo "Created working directory: $WORKING_DIR"

# Create subdirectories
mkdir -p "$WORKING_DIR/config_for_mip_start"
mkdir -p "$WORKING_DIR/expansion"
mkdir -p "$WORKING_DIR/mip_start"
mkdir -p "$WORKING_DIR/param_ils"
mkdir -p "$WORKING_DIR/param_ils/parameter"
mkdir -p "$WORKING_DIR/param_ils/scenario"
mkdir -p "$WORKING_DIR/pruning"
mkdir -p "$WORKING_DIR/pruning/input"
mkdir -p "$WORKING_DIR/pruning/output"
mkdir -p "$WORKING_DIR/solver"
mkdir -p "$WORKING_DIR/solver/outfiles"
mkdir -p "$WORKING_DIR/solver/mipstarts"
echo "Created subdirectories in: $WORKING_DIR"