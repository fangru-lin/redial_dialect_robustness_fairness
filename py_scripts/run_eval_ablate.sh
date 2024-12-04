#!/bin/bash

# Check if at least one argument is provided
if [ -z "$1" ]; then
  echo "Usage: $0 [additional arguments]"
  exit 1
fi

for DENSITY in 0.06 0.04 0.02; do
  COMMON_ARGS="--no_cot True --no_aave_test True --n_choices 1 --temperature 0 --output_dir ../outputs/typo"
  CODE_INPUT="--input_path ../data/redial/perturbations/typo/$DENSITY/algorithm.json"
  LOGIC_INPUT="--input_path ../data/redial/perturbations/typo/$DENSITY/logic.json"
  MATH_INPUT="--input_path ../data/redial/perturbations/typo/$DENSITY/math.json"
  COMPREHENSIVE_INPUT="--input_path ../data/redial/perturbations/typo/$DENSITY/comprehensive.json"

  # Run each Python script with the provided arguments
  python3 eval_code.py $COMMON_ARGS/$DENSITY $CODE_INPUT "$@"
  python3 eval_logic.py $COMMON_ARGS/$DENSITY $LOGIC_INPUT "$@"
  python3 eval_math.py $COMMON_ARGS/$DENSITY $MATH_INPUT "$@"
  python3 eval_comprehensive.py $COMMON_ARGS/$DENSITY $COMPREHENSIVE_INPUT "$@"
done

COMMON_ARGS="--no_cot True --n_choices 1 --temperature 0 --sae_ablate True --output_dir ../outputs/sae_ablate"

# Run each Python script with the provided arguments
python3 eval_code.py $COMMON_ARGS "$@"
python3 eval_logic.py $COMMON_ARGS "$@"
python3 eval_math.py $COMMON_ARGS "$@"
python3 eval_comprehensive.py $COMMON_ARGS "$@"

for DENSITY in 1 0.75 0.5 0.25; do
  COMMON_ARGS="--no_cot True --no_aave_test True --n_choices 1 --temperature 0 --output_dir ../outputs/multivalue"
  CODE_INPUT="--input_path ../data/redial/perturbations/multivalue/$DENSITY/algorithm.json"
  LOGIC_INPUT="--input_path ../data/redial/perturbations/multivalue/$DENSITY/logic.json"
  MATH_INPUT="--input_path ../data/redial/perturbations/multivalue/$DENSITY/math.json"
  COMPREHENSIVE_INPUT="--input_path ../data/redial/perturbations/multivalue/$DENSITY/comprehensive.json"

  # Run each Python script with the provided arguments
  python3 eval_code.py $COMMON_ARGS/$DENSITY $CODE_INPUT "$@"
  python3 eval_logic.py $COMMON_ARGS/$DENSITY $LOGIC_INPUT "$@"
  python3 eval_math.py $COMMON_ARGS/$DENSITY $MATH_INPUT "$@"
  python3 eval_comprehensive.py $COMMON_ARGS/$DENSITY $COMPREHENSIVE_INPUT "$@"
done