export MODEL_ID="runwayml/stable-diffusion-v1-5"
export DATASET_ID="Jephson/edited-sky-dataset-4"
export OUTPUT_DIR="addsky-train-4"

python -c "from huggingface_hub.hf_api import HfFolder; HfFolder.save_token('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')"

accelerate launch --mixed_precision="fp16" train_instruct_pix2pix.py \
  --pretrained_model_name_or_path=$MODEL_ID \
  --dataset_name=$DATASET_ID \
  --use_ema \
  --enable_xformers_memory_efficient_attention \
  --resolution=64 --random_flip \
  --train_batch_size=8 --gradient_accumulation_steps=4 --gradient_checkpointing \
  --max_train_steps=2100 \
  --checkpointing_steps=350 --checkpoints_total_limit=10 \
  --learning_rate=2e-04 --lr_warmup_steps=0 \
  --mixed_precision=fp16 \
  --val_image_url="https://hf.co/datasets/diffusers/diffusers-images-docs/resolve/main/mountain.png" \
  --validation_prompt="Add sky for this photo" \
  --seed=42 \
  --output_dir=$OUTPUT_DIR \
  --report_to=wandb \
  --push_to_hub
