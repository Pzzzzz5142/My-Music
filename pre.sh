fairseq-preprocess \
    --only-source \
    --trainpref train.train.tokens \
    --validpref train.valid.tokens \
    --testpref train.test.tokens \
    --destdir data-bin/remi_midi --workers 20