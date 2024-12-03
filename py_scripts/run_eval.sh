#!/bin/bash

# Check if at least one argument is provided
if [ -z "$1" ]; then
  echo "Usage: $0 [additional arguments]"
  exit 1
fi

COMMON_ARGS=" --n_choices 1 --temperature 0 --output_dir ../outputs"
CODE_INPUT="--input_path ../data/redial/redial_gold/algorithm.json"
LOGIC_INPUT="--input_path ../data/redial/redial_gold/logic.json"
MATH_INPUT="--input_path ../data/redial/redial_gold/math.json"
COMPREHENSIVE_INPUT="--input_path ../data/redial/redial_gold/comprehensive.json"

# Run each Python script with the provided arguments
python3 eval_code.py $COMMON_ARGS "$@"
python3 eval_logic.py $COMMON_ARGS "$@"
python3 eval_math.py $COMMON_ARGS "$@"
python3 eval_comprehensive.py $COMMON_ARGS "$@"