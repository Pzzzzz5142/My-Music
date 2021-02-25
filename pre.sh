fairseq-preprocess \
    --only-source \
    --trainpref data/maecl.train.tokens \
    --validpref data/maecl.valid.tokens \
    --testpref data/maecl.test.tokens \
    --destdir data-bin/cl --workers 20