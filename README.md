# Env

trt-llm based on b171e87 (https://github.com/NVIDIA/TensorRT-LLM.git)

modelopt version：0.19.0

# Cmd:

FP16:
```
python convert_checkpoint.py --model_dir /root/.cache/huggingface/hub/models--amd--AMD-OLMo-1B-SFT/snapshots/4c54fc6babb00e7e71a724e13ec9b3ec6f08266e/ --output_dir ./tllm_checkpoint_1gpu_fp16 --dtype float16

trtllm-build --checkpoint_dir ./tllm_checkpoint_1gpu_fp16             --output_dir ./fp16_engine             --gemm_plugin auto

nsys profile -o fp16_olmo ../../cpp/build/benchmarks/gptSessionBenchmark --engine_dir ./fp16_engine/ --batch_size 8 --input_output_len 1024,8 --warm_up 1 --num_runs 2 --duration 0

python ../mmlu.py --test_trt_llm --engine_dir ./fp16_engine/ --tokenizer_dir /root/.cache/huggingface/hub/models--amd--AMD-OLMo-1B-SFT/snapshots/4c54fc6babb00e7e71a724e13ec9b3ec6f08266e/  --data_dir /fp8_allreduce/data/data/
```
Average mmlu accuracy: 0.293


FP8:
```
python ../quantization/quantize.py --model_dir /root/.cache/huggingface/hub/models--amd--AMD-OLMo-1B-SFT/snapshots/4c54fc6babb00e7e71a724e13ec9b3ec6f08266e/ \
                                   --dtype float16 \
                                   --qformat fp8 \
                                   --kv_cache_dtype fp8 \
                                   --output_dir ./tllm_checkpoint_1gpu_fp8 \
                                   --calib_size 512
                                   
trtllm-build --checkpoint_dir ./tllm_checkpoint_1gpu_fp8             --output_dir ./fp8_engine             --gemm_plugin auto

nsys profile -o fp8_olmo ../../cpp/build/benchmarks/gptSessionBenchmark --engine_dir ./fp8_engine/ --batch_size 8 --input_output_len 1024,8 --warm_up 1 --num_runs 2 --duration 0

python ../mmlu.py --test_trt_llm --engine_dir ./fp8_engine/ --tokenizer_dir /root/.cache/huggingface/hub/models--amd--AMD-OLMo-1B-SFT/snapshots/4c54fc6babb00e7e71a724e13ec9b3ec6f08266e/  --data_dir /fp8_allreduce/data/data/
```
Average mmlu accuracy: 0.285

# Performance fp8 vs fp16
input1024 ouput8 bs8

context phase: 1.6x speed up

generation phase: 1.4x speed up
