#!/usr/bin/env bash
for x in exp/*/*_decode; do [ -d $x ] && grep WER $x/cer_* | utils/best_wer.sh; done 2>/dev/null
