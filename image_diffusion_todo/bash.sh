conda activate ddpm
cd  /data/XHH/DDPM/Lab1-DDPM/image_diffusion_todo
python train.py --mode linear --predictor noise
python train.py --mode linear --predictor x0
python train.py --mode linear --predictor x0
python train.py --mode quad --predictor noise
python train.py --mode quad --predictor x0
python sample.py --mode linear --predictor noise
python sample.py --mode linear --predictor x0
python sample.py --mode quad --predictor noise
python sample.py --mode quad --predictor x0
