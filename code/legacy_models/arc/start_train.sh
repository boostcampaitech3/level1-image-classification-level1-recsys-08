SM_CHANNEL_TRAIN=/opt/ml/input/data/train/images SM_MODEL_DIR=. python train.py --lr 0.002 --epochs 1 \
	--model CustomModel_Arc \
	--optimizer Adam \
	--epochs 30


