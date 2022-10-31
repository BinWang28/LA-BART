#!/bin/bash
#SBATCH --output=./1.log
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --partition=new
##SBATCH --exclude=hlt06,ttnusa9,ttnusa10


echo "= = = = = = = = = = = = = ="
echo "The project is running..."
currentDate=`date`
echo $currentDate
start=`date +%s`
echo "= = = = = = = = = = = = = ="

python train.py \
    --len_input 'no' \
    --len_output 'no' \
    --output_dir ./output/1 \
    --train_file ./data/dialogsum/dialogsum.train.jsonl \
    --validation_file ./data/dialogsum/dialogsum.dev.jsonl \
    --test_file ./data/dialogsum/dialogsum.test.jsonl \
    --text_column dialogue \
    --summary_column summary \
    --model_name_or_path facebook/bart-large \
    --model_type bart \
    --max_source_length 1024 \
    --min_target_length 1 \
    --max_target_length 128 \
    --num_beams 4 \
    --learning_rate 5e-5 \
    --weight_decay 1e-3 \
    --label_smoothing 0.1 \
    --length_penalty 1.0 \
    --num_train_epochs 15 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 64 \
    --per_device_eval_batch_size 8 \
    --per_device_test_batch_size 8 \
    --num_warmup_steps 0 \
    --cache_dir ./output/cache \
    --overwrite_cache True \
    --seed 12345

echo "= = = = = = = = = = = = = ="
echo "The project is Finished..."
end=`date +%s`
runtime=$((end-start))
echo "The program takes '$((runtime / 60))' minutes."
echo "= = = = = = = = = = = = = ="
