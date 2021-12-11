rm -rf model_structured
mkdir -p model_structured
cd model_structured || exit

FORMAT=structured
TOTAL_NUM_UPDATES=9210
WARMUP_UPDATES=553
LR=1e-05
HEAD_NAME=decode_${FORMAT}_head
NUM_CLASSES=2
MAX_SENTENCES=32
ROBERTA_PATH=../roberta.base/model.pt

fairseq-train ../decode-bin/$FORMAT/ \
  --restore-file $ROBERTA_PATH \
  --max-positions 512 \
  --batch-size $MAX_SENTENCES \
  --max-tokens 4400 \
  --task sentence_prediction \
  --reset-optimizer --reset-dataloader --reset-meters \
  --required-batch-size-multiple 1 \
  --init-token 0 --separator-token 2 \
  --arch roberta_base \
  --criterion sentence_prediction \
  --classification-head-name $HEAD_NAME \
  --num-classes $NUM_CLASSES \
  --dropout 0.1 --attention-dropout 0.1 \
  --weight-decay 0.1 --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 \
  --clip-norm 0.0 \
  --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
  --max-epoch 10 \
  --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric \
  --shorten-method "truncate" \
  --find-unused-parameters \
  --update-freq 1
