fairseq-preprocess \
    --only-source \
    --trainpref data/mae.train.tokens \
    --validpref data/mae.valid.tokens \
    --testpref data/mae.test.tokens \
    --destdir data-bin/new --workers 20