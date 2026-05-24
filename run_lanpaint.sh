#!/usr/bin/env bash
# Unified LanPaint CLI examples
# Usage: uncomment one command block, or copy and customize it.

# Launch the Streamlit UI (recommended)
# uv run streamlit run app.py

# List all registered models
# python run_lanpaint.py --list-models

# Flux2 Klein (URL example)
# python run_lanpaint.py --model flux-klein \
#     --prompt "change building's window light color to blue" \
#     --image "https://raw.githubusercontent.com/scraed/LanPaint/master/examples/Example_24/Original_No_Mask.png" \
#     --mask "https://raw.githubusercontent.com/scraed/LanPaint/master/examples/Example_24/Masked_Load_Me_in_Loader.png"

# Flux2 Klein (local paths)
uv run python run_lanpaint.py --model flux-klein \
    --model-id /home/aman/.cache/huggingface/hub/models--black-forest-labs--FLUX.2-klein-4B/snapshots/e7b7dc27f91deacad38e78976d1f2b499d76a294 \
    --local-files-only \
    --prompt "Generate broken fender on image1 at black region. Do not change the color of the car in image1. Do not generate damage in other regions of image1" \
    --image "/home/aman/Documents/code/image_gen/Qwen_2511_Flux0.2/data/org_2022_Honda_City_ZX_i-VTEC (Copy).jpg" \
    --mask "/home/aman/Documents/code/image_gen/Qwen_2511_Flux0.2/data/org_2022_Honda_City_ZX_i-VTEC_bumper_black.jpg"

# SD3
# python run_lanpaint.py --model sd3 \
#     --lp-n-steps 5 \
#     --guidance-scale 5.5 \
#     --num-steps 30 \
#     --prompt "a bottle with a rainbow galaxy inside it on top of a wooden table on a snowy mountain top with the ocean and clouds in the background" \
#     --image "https://raw.githubusercontent.com/scraed/LanPaint/master/examples/Example_9/Original_No_Mask.png" \
#     --mask "https://raw.githubusercontent.com/scraed/LanPaint/master/examples/Example_9/Masked_Load_Me_in_Loader.png"

# Z-Image Turbo Inpaint
# python run_lanpaint.py --model z-image \
#     --lp-n-steps 5 \
#     --lp-friction 15.0 \
#     --lp-lambda 16 \
#     --seed 0 \
#     --guidance-scale 1.0 \
#     --num-steps 9 \
#     --prompt "Latina female with thick wavy hair, white shirt, harbor boats and pastel houses behind. Breezy seaside light, warm tones, cinematic close-up." \
#     --image "https://raw.githubusercontent.com/scraed/LanPaint/master/examples/Example_21/Original_No_Mask.png" \
#     --mask "https://raw.githubusercontent.com/scraed/LanPaint/master/examples/Example_21/Masked_Load_Me_in_Loader.png"

# Z-Image Turbo Outpaint
# python run_lanpaint.py --model z-image \
#     --lp-n-steps 5 \
#     --lp-friction 15.0 \
#     --lp-lambda 16 \
#     --seed 42 \
#     --guidance-scale 1.0 \
#     --num-steps 15 \
#     --prompt "Latina female with thick wavy hair, white shirt, harbor boats and pastel houses behind. Breezy seaside light, warm tones, cinematic close-up." \
#     --image "https://raw.githubusercontent.com/scraed/LanPaint/master/examples/Example_22/Original_No_Mask.png" \
#     --outpaint-pad "l200r200t200b200" \

# # Qwen Image Edit Inpaint
# python run_lanpaint.py --model qwen \
#     --prompt "change the girl's cloth to red evening gown" \
#     --image "https://raw.githubusercontent.com/scraed/LanPaint/master/examples/Example_14/Original_No_Mask.png" \
#     --mask "https://raw.githubusercontent.com/scraed/LanPaint/master/examples/Example_14/Masked_Load_Me_in_Loader.png" \
#     --seed 0 \
#     --num-steps 20 \
#     --guidance-scale 2.5 \
#     --lp-n-steps 5