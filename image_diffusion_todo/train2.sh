conda activate ddpm
cd  /data/XHH/DDPM/Lab1-DDPM/image_diffusion_todo
#python train.py --mode linear --predictor noise
#python train.py --mode linear --predictor x0
#python train.py --mode quad --predictor noise
#python train.py --mode quad --predictor x0
export CUDA_VISIBLE_DEVICES=5
#python train.py --mode cosine --predictor noise
#python train.py --mode cosine --predictor x0

#python train.py --mode linear --predictor mean
#python train.py --mode cosine --predictor mean
#python train.py --mode quad --predictor mean


--mode: linear, quad, cosine
--predictor: noise, x0, mean

python train.py --mode linear --predictor x0
python train.py --mode quad --predictor noise

python sample.py --mode linear --predictor noise
python sample.py --mode linear --predictor x0
python sample.py --mode quad --predictor noise
python sample.py --mode quad --predictor x0
