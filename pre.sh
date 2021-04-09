fairseq-preprocess \
    --only-source \
    --trainpref data/mae_remi.train.tokens \
    --validpref data/mae_remi.valid.tokens \
    --testpref data/mae_remi.test.tokens \
    --destdir data-bin/mae_remi --workers 20