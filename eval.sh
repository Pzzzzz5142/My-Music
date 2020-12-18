fairseq-eval-lm data-bin/maestro-v2.0.0\
    --path checkpoints/transformer_music/checkpoint_best.pt \
    --batch-size 2 \
    --tokens-per-sample 511 \
    --context-window 400