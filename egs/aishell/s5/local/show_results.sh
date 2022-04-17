# !/bin/bash
for x in exp/*/decode_test; do [ -d $x ] && grep WER $x/cer_* | utils/best_wer.sh; done 2>/dev/null
for x in exp/nnet3/*/decode_test; do [ -d $x ] && grep WER $x/cer_* | utils/best_wer.sh; done 2>/dev/null
for x in exp/*/*_test_decode; do [ -d $x ] && grep WER $x/cer_* | utils/best_wer.sh; done 2>/dev/null
