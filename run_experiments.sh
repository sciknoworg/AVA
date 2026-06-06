!/bin/bash

 MARGINS=(0.3 0.5 0.7)

 for MARGIN in "${MARGINS[@]}"; do
     echo ""
     echo "================================================================="
     echo "Running experiment with margin: $MARGIN"
     echo "================================================================="

     python finetune_triplet.py \
          --train_path      data/train.json \
          --model_id        sentence-transformers/all-MiniLM-L6-v2 \
          --output_dir      "assets/MiniLM-triplet-m${MARGIN}" \
          --num_train_epochs 10 \
          --per_device_train_batch_size 256 \
          --per_device_eval_batch_size 256 \
          --margin "${MARGIN}" \
          --fp16
 done

 echo "All experiments completed successfully!"


HYPERBOLIC_Cs=(0.1 0.3 0.5 1.0)
for HYPERBOLIC_C in "${HYPERBOLIC_Cs[@]}"; do
    echo ""
    echo "================================================================="
    echo "Running experiment with hyperbolic c: $HYPERBOLIC_C"
    echo "================================================================="

    python finetune_triplet.py \
         --train_path      data/train.json \
         --model_id        sentence-transformers/all-MiniLM-L6-v2 \
         --output_dir      "assets/MiniLM-triplet-hyperbolic-m0.3-c${HYPERBOLIC_C}" \
         --num_train_epochs 10 \
         --per_device_train_batch_size 256 \
         --per_device_eval_batch_size 256 \
         --margin 0.3 \
         --hyperbolic_c "${HYPERBOLIC_C}" \
         --fp16 \
         --hyperbolic
done

echo "All experiments completed successfully!"



DPO_BETAs=(0.1 0.3 0.5)
for RL_BETA in "${DPO_BETAs[@]}"; do
    echo ""
    echo "================================================================="
    echo "Running experiment with DPO beta: $RL_BETA"
    echo "================================================================="

    python finetune_triplet.py \
         --train_path      data/train.json \
         --model_id        sentence-transformers/all-MiniLM-L6-v2 \
         --output_dir      "assets/MiniLM-triplet-dpo-m0.3-beta${RL_BETA}" \
         --num_train_epochs 10 \
         --per_device_train_batch_size 256 \
         --per_device_eval_batch_size 256 \
         --margin 0.3 \
         --rl_beta "${RL_BETA}" \
         --fp16 \
         --dpo
done

echo "All experiments completed successfully!"

echo "================================================================="
echo "Running experiment with Hyperbolic + DPO"
echo "================================================================="
python finetune_triplet.py \
        --train_path      data/train.json \
        --model_id        sentence-transformers/all-MiniLM-L6-v2 \
        --output_dir      "assets/MiniLM-triplet-hyperbolic-dpo-m0.3-c0.3-beta0.5-w_dpo0.3-w_hyperbolic0.7" \
        --num_train_epochs 10 \
        --per_device_train_batch_size 256 \
        --per_device_eval_batch_size 256 \
        --margin 0.3 \
        --rl_beta 0.5 \
        --hyperbolic_c 0.3\
        --fp16 \
        --dpo \
        --hyperbolic \
        --w_hyperbolic 0.7\
        --w_dpo 0.3

echo "All experiments completed successfully!"

echo "================================================================="
echo "Running Evaluations"
echo "================================================================="

python evaluate.py