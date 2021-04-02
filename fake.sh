fairseq-train --task language_modeling \
  data-bin/fake_512 \
  --save-dir /mnt/zhangyi/checkpoints/transformer_music_fake_512\
  --arch transformer_lm \
  --optimizer adam --adam-betas '(0.9, 0.98)' --weight-decay 0.01 --clip-norm 0.0 \
  --lr 0.0007 --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07 \
  --tokens-per-sample 2048 --sample-break-mode none \
  --max-tokens 4096 --update-freq 1 \
  --max-update 100000 \
  --future-target 