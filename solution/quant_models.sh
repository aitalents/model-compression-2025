#!/bin/bash

optimum-cli export onnx --model svalabs/twitter-xlm-roberta-crypto-spam twitter_xlm_roberta_base_sentiment_onnx
optimum-cli onnxruntime quantize --onnx_model twitter_xlm_roberta_base_sentiment_onnx/ --avx512 -o twitter_xlm_roberta_base_sentiment_onnx_quantized
rm -rf twitter_xlm_roberta_base_sentiment_onnx

optimum-cli export onnx --model ivanlau/language-detection-fine-tuned-on-xlm-roberta-base language_detection_fine_tuned_on_xlm_roberta_base_onnx
optimum-cli onnxruntime quantize --onnx_model language_detection_fine_tuned_on_xlm_roberta_base_onnx/ --avx512 -o language_detection_fine_tuned_on_xlm_roberta_base_onnx_quantized
rm -rf language_detection_fine_tuned_on_xlm_roberta_base_onnx

optimum-cli export onnx --model svalabs/twitter-xlm-roberta-crypto-spam twitter_xlm_roberta_crypto_spam_onnx
optimum-cli onnxruntime quantize --onnx_model twitter_xlm_roberta_crypto_spam_onnx/ --avx512 -o twitter_xlm_roberta_crypto_spam_onnx_quantized
rm -rf twitter_xlm_roberta_crypto_spam_onnx

optimum-cli export onnx --model EIStakovskii/xlm_roberta_base_multilingual_toxicity_classifier_plus xlm_roberta_base_multilingual_toxicity_classifier_plus_onnx
optimum-cli onnxruntime quantize --onnx_model xlm_roberta_base_multilingual_toxicity_classifier_plus_onnx/ --avx512 -o xlm_roberta_base_multilingual_toxicity_classifier_plus_onnx_quantized
rm -rf xlm_roberta_base_multilingual_toxicity_classifier_plus_onnx

optimum-cli export onnx --model jy46604790/Fake-News-Bert-Detect fake_news_bert_detect_onnx
optimum-cli onnxruntime quantize --onnx_model fake_news_bert_detect_onnx/ --avx512 -o fake_news_bert_detect_onnx_quantized
rm -rf fake_news_bert_detect_onnx
