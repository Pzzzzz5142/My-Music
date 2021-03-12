fairseq-preprocess \
    --source-lang en --target-lang de \
    --trainpref data/mae.train.tokens \
    --validpref data/mae.valid.tokens \
    --testpref data/mae.test.tokens \
    --destdir data-bin/autoencoder --workers 20