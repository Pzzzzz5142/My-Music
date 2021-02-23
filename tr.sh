fairseq-train --task language_modeling \
  data-bin/maestro-v2.0.0 \
  --save-dir /mnt/zhangyi/checkpoints/transformer_music \
  --arch music_transformer --share-decoder-input-output-embed \
  --dropout 0.1 \
  --optimizer adam --adam-betas '(0.9, 0.98)' --weight-decay 0.01 --clip-norm 0.0 \
  --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07 \
  --tokens-per-sample 2048 --sample-break-mode none --shorten-method random_crop \
  --max-tokens 4096 --update-freq 16 \
  --max-update 50000 \
  --future-target \
  --fp16 
