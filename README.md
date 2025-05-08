# guided-diffusion

This is the **re-implementation version** for [Diffusion Models Beat GANS on Image Synthesis](http://arxiv.org/abs/2105.05233).

This repository is based on [openai/improved-diffusion](https://github.com/openai/improved-diffusion), with modifications for classifier conditioning and architecture improvements.

# Environment Configuration

```
python == 3.8

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 （PyTorch 2.0+，支持NVIDIA A100 GPU）

pip install -e .  （安装Guided diffusion依赖的包）

conda install conda-forge::mpi4py
```

# Training (using LSUN-Bedroom as an example)

```
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export MODEL_FLAGS="--image_size 256 --num_channels 128 --num_res_blocks 3 --learn_sigma True"

export DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule cosine --use_kl True"

（--lr_anneal_steps这里设置为图片总数量/batch_size，图片总数量最好为batch_size的整数倍，这样DataLoader不会丢弃最后一个batch）
export TRAIN_FLAGS="--lr 1e-4 --batch_size 8 --lr_anneal_steps 3750 --schedule_sampler loss-second-moment"

export OPENAI_LOGDIR="/project/wanruibo/PaperCode/guided-diffusion/models/Mymodel"

python scripts/image_train.py --data_dir lsun_train_output_dir $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS
```

# Sampling

## LSUN-Bedroom

```
export SAMPLE_FLAGS="--batch_size 4 --num_samples 5000 --timestep_respacing 250"

export MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --dropout 0.1 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"

export OPENAI_LOGDIR="/project/wanruibo/PaperCode/guided-diffusion/results/sample_bedroom"

python image_sample.py $MODEL_FLAGS --model_path models/lsun_bedroom.pt $SAMPLE_FLAGS
```

## 256x256-Classifier

```
export SAMPLE_FLAGS="--batch_size 4 --num_samples 5000 --timestep_respacing 250"

export MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"

export OPENAI_LOGDIR="/project/wanruibo/PaperCode/guided-diffusion/results/sample_256x256_classifier"

python classifier_sample.py $MODEL_FLAGS --classifier_scale 1.0 --classifier_path models/256x256_classifier.pt --model_path models/256x256_diffusion.pt $SAMPLE_FLAGS
```

# Results

## LSUN-Bedroom 

```
cd evaluations

pip install -r requirements

python evaluations/evaluator.py /project/wanruibo/PaperCode/guided-diffusion/results/ref_dataset/VIRTUAL_lsun_bedroom256.npz /project/wanruibo/PaperCode/guided-diffusion/results/sample_bedroom/samples_100x256x256x3.npz
```

| Dataset               | FID  | Precision | Recall |
|:---------------------:|:----:|:---------:|:------:|
| LSUN-Bedroom          | 4.85 |    0.65   | 0.51   |
| LSUN-Bedroom（原论文） | 1.90 |    0.66   | 0.51   |

## 256x256-Classifier

```
python evaluations/evaluator.py /project/wanruibo/PaperCode/guided-diffusion/results/ref_dataset/VIRTUAL_imagenet256_labeled.npz /project/wanruibo/PaperCode/guided-diffusion/results/sample_256x256_classifier/samples_5000x256x256x3.npz
```

| Dataset                   | FID   | Precision | Recall |
|:-------------------------:|:-----:|:---------:|:------:|
| ImageNet 256x256          | 11.20 |    0.82   | 0.68   |
| ImageNet 256x256（原论文） | 4.59  |    0.82   | 0.52   |



