# Few-shot Learning Model Edge Implement

## Hardware environment
- Device: OPPO RENO Z
- CPU: MTK P90

## How to reproduce
- Run smart phone as AIoT device
```shell
adb shell /data/local/tmp/ClassifyImage \
        --count 1 \
        --profiling 1 \
        --verbose 0 \
        --num_results 1 \
        --tflite_model /data/local/tmp/quantized_model.tflite \
        --labels /data/local/tmp/dropt/labels \
        --image_bin /data/local/tmp/torch_image_uint8/ \
        --input_mean 0.5 \
        --input_std 0.5
```
