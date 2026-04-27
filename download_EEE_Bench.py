from datasets import load_dataset
import os
import json
from PIL import Image

ds = load_dataset("afdsafas/EEE-Bench")

# Create folders
os.makedirs("EEE_Bench/images", exist_ok=True)
os.makedirs("EEE_Bench/qa", exist_ok=True)

all_qa = []

# Loop through all splits (train/test/validation)
for split_name, split_data in ds.items():
    print(f"Saving split: {split_name} ({len(split_data)} samples)")
    
    for i, sample in enumerate(split_data):
        
        # Save image if it exists
        image_filename = f"{split_name}_{i}.png"
        if "image" in sample and sample["image"] is not None:
            img = sample["image"]
            if isinstance(img, Image.Image):
                img.save(f"EEE_Bench/images/{image_filename}")

        # Save Q&A
        all_qa.append({
            "id":       f"{split_name}_{i}",
            "split":    split_name,
            "image":    image_filename,
            "question": sample.get("problem", ""),   # ✅ correct key
            "answer":   sample.get("solution", ""),  # ✅ correct key
        })

# Save all Q&A to JSON
with open("EEE_Bench/qa/eee_bench_qa.json", "w") as f:
    json.dump(all_qa, f, indent=2)

print(f"\n✅ Done! Saved {len(all_qa)} Q&A pairs")
print(f"   Images → EEE_Bench/images/")
print(f"   Q&A    → EEE_Bench/qa/eee_bench_qa.json")