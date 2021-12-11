PREFIX=decode_v0.1/formatted

# BPE encode
for FORMAT in structured unstructured; do
  for SPLIT in train dev; do
    python3 -m multiprocessing_bpe_encoder \
      --encoder-json encoder.json \
      --vocab-bpe vocab.bpe \
      --inputs "$PREFIX/$FORMAT/$SPLIT.input0" "$PREFIX/$FORMAT/$SPLIT.input1" \
      --outputs "$PREFIX/$FORMAT/$SPLIT.input0.bpe" "$PREFIX/$FORMAT/$SPLIT.input1.bpe" \
      --workers 16 \
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
      --workers 16 \
      --srcdict dict.txt
  done
  fairseq-preprocess \
    --only-source \
    --trainpref "$PREFIX/$FORMAT/train.label" \
    --validpref "$PREFIX/$FORMAT/dev.label" \
    --destdir "decode-bin/$FORMAT/label" \
    --workers 16
done
