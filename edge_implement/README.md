# Few-shot Learning Model Edge Implement

## Hardware environment
- Device: OPPO RENO Z
- CPU: MTK P90

## How to reproduce
- Generate calibration `.npy` from training images
```shell
python3 ./data/gen_data.py --ftype npy
```
- Generate test binary data from test images
```shell
python3 ./data/gen_data.py --ftype bin --dtype uint
```
- Convert model
```shell
python3 ./model/convert_resnet.py --class_num 5 --input_shape 1,3,224,224 --pretrained PRETRAINED_MODEL_PATH --target DUMP_MODEL_PATH
```
- Run smart phone as AIoT device
```shell
adb shell /data/local/tmp/ClassifyImage \
        --count 1 \
        --profiling 1 \
        --verbose 0 \
        --num_results 1 \
        --tflite_model /data/local/tmp/__[ QUANTIZED_MODEL | FLOAT32_MODEL ].tflite__ \
        --labels /data/local/tmp/__LABELS__ \
        --image_bin /data/local/tmp/__IMAGE_BIN_DIR__ \
        --input_mean 0.5 \
        --input_std 0.5
```

## Test result
### Comparison
| Quantization | Before | After |
|:------------ | ------ |:----- |
| FPS (Invoke) | 21     | 27    |
| Model Size   | 43M    | 11M   |
### Details
- Quantized model
![](https://i.imgur.com/xMkVjuZ.png)
- FP32 model
![](https://i.imgur.com/US8Basm.png)

## Reference
- Our binary executable file is built based on our [previous work](https://github.com/Deadline-Driven/NASA2019_Project) and MTK NeuroPilot SDK
