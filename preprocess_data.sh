# DECODE dataset
PREFIX=decode_v0.1/formatted

# BPE encode
for FORMAT in structured unstructured; do
  for SPLIT in train dev; do
    python3 -m multiprocessing_bpe_encoder \
      --encoder-json encoder.json \
      --vocab-bpe vocab.bpe \
      --inputs "$PREFIX/$FORMAT/$SPLIT.input0" "$PREFIX/$FORMAT/$SPLIT.input1" \
      --outputs "$PREFIX/$FORMAT/$SPLIT.input0.bpe" "$PREFIX/$FORMAT/$SPLIT.input1.bpe" \
      --workers 60 \
      --keep-empty
  done
done

rm -rf decode-bin

# Preprocess data
for FORMAT in structured unstructured; do
  for idx in 0 1; do
    fairseq-preprocess \
      --only-source \
      --trainpref "$PREFIX/$FORMAT/train.input$idx.bpe" \
      --validpref "$PREFIX/$FORMAT/dev.input$idx.bpe" \
      --destdir "decode-bin/$FORMAT/input$idx" \
      --workers 60 \
      --srcdict dict.txt
  done
  fairseq-preprocess \
    --only-source \
    --trainpref "$PREFIX/$FORMAT/train.label" \
    --validpref "$PREFIX/$FORMAT/dev.label" \
    --destdir "decode-bin/$FORMAT/label" \
    --workers 60
done

# ANLI-R3 dataset
PREFIX=anli_v1.0/R3/formatted

# BPE encode
for SPLIT in train dev; do
  python3 -m multiprocessing_bpe_encoder \
    --encoder-json encoder.json \
    --vocab-bpe vocab.bpe \
    --inputs "$PREFIX/$SPLIT.input0" "$PREFIX/$SPLIT.input1" \
    --outputs "$PREFIX/$SPLIT.input0.bpe" "$PREFIX/$SPLIT.input1.bpe" \
    --workers 60 \
    --keep-empty
done

rm -rf anli-r3-bin

# Preprocess data
for idx in 0 1; do
  fairseq-preprocess \
    --only-source \
    --trainpref "$PREFIX/train.input$idx.bpe" \
    --validpref "$PREFIX/dev.input$idx.bpe" \
    --destdir "anli-r3-bin/input$idx" \
    --workers 60 \
    --srcdict dict.txt
done
fairseq-preprocess \
  --only-source \
  --trainpref "$PREFIX/train.label" \
  --validpref "$PREFIX/dev.label" \
  --destdir "anli-r3-bin/label" \
  --workers 60
