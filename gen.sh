fairseq-interactive data-bin/maestro-v2.0.0 \
--task language_modeling \
--path checkpoints/transformer_music/checkpoint_best.pt \
--sampling --beam 1 --sampling-topk 10