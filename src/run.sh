#!/bin/bash

export LC_ALL=C.UTF-8
export LANG=C.UTF-8

export OUTPUT_DIR=/home/m3hrdadfi/code/t5-recipe-generation
export MODEL_NAME_OR_PATH=t5-base
# export MODEL_NAME_OR_PATH=flax-community/t5-recipe-generation
export NUM_BEAMS=3

export TRAIN_FILE=/home/m3hrdadfi/code/data/train.csv
export VALIDATION_FILE=/home/m3hrdadfi/code/data/test.csv
export TEST_FILE=/home/m3hrdadfi/code/data/test.csv
export TEXT_COLUMN=inputs
export TARGET_COLUMN=targets
export MAX_SOURCE_LENGTH=256
export MAX_TARGET_LENGTH=1024
export SOURCE_PREFIX=items
export MAX_EVAL_SAMPLES=5000

export PER_DEVICE_TRAIN_BATCH_SIZE=8
export PER_DEVICE_EVAL_BATCH_SIZE=8
export GRADIENT_ACCUMULATION_STEPS=2
export NUM_TRAIN_EPOCHS=5.0
export LEARNING_RATE=5e-4
export WARMUP_STEPS=5000
export LOGGING_STEPS=500
export EVAL_STEPS=2500
export SAVE_STEPS=2500

python src/run_recipe_nlg_flax.py \
    --output_dir="$OUTPUT_DIR"  \
    --train_file="$TRAIN_FILE" \
    --validation_file="$VALIDATION_FILE" \
    --max_eval_samples=$MAX_EVAL_SAMPLES \
    --text_column="$TEXT_COLUMN" \
    --target_column="$TARGET_COLUMN" \
    --source_prefix="$SOURCE_PREFIX: " \
    --max_source_length="$MAX_SOURCE_LENGTH" \
    --max_target_length="$MAX_TARGET_LENGTH" \
    --model_name_or_path="$MODEL_NAME_OR_PATH"  \
    --extra_tokens="" \
    --special_tokens="<sep>,<section>" \
    --per_device_train_batch_size=$PER_DEVICE_TRAIN_BATCH_SIZE \
    --per_device_eval_batch_size=$PER_DEVICE_EVAL_BATCH_SIZE \
    --gradient_accumulation_steps=$GRADIENT_ACCUMULATION_STEPS \
    --num_train_epochs=$NUM_TRAIN_EPOCHS \
    --learning_rate=$LEARNING_RATE \
    --warmup_steps=$WARMUP_STEPS \
    --logging_step=$LOGGING_STEPS \
    --eval_steps=$EVAL_STEPS \
    --save_steps=$SAVE_STEPS \
    --prediction_debug \
    --do_train \
    --do_eval \
    --overwrite_output_dir \
    --predict_with_generate \
    --push_to_hub
