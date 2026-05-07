#!/bin/bash
echo "🚀 Running example SSL experiment..."
python3 scripts/quick_start.py \
    --labeled_samples 1000 \
    --epochs 10 \
    --batch_size 32 \
    --lr 0.001 \
    --ssl_method pseudo_labeling \
    --output_dir example_output

echo "✅ Example experiment completed!"
echo "📊 Results saved to example_output/"
echo "🎯 To view results: python3 scripts/evaluate.py --experiment_dir example_output"
