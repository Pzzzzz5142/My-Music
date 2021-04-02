fairseq-preprocess \
    --only-source \
    --trainpref fake_one_beat_512.tokens.train \
    --validpref fake_one_beat_512.tokens.valid \
    --testpref fake_one_beat_512.tokens.test \
    --destdir data-bin/fake_512 --workers 20