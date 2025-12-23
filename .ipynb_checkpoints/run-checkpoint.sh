torchrun --standalone --nproc_per_node=8 nano_cli.py \
--num-devices 8 --batch-size 512 --device-batch-size 64 --num-iterations 5100 \
--warmup-steps 2000 --min-lr 0.0 \
--opt1 AdamWScheduleFree \
--opt1-kwargs lr=0.008,betas=[0.95,0.99],eps=1e-8,weight_decay=0.0 \
--opt2 NorMuonScheduleFree \
--opt2-kwargs lr=0.008,betas=[0.9,0.95],eps=1e-8,eta_scale=0.2,backend=newtonschulz5,backend_steps=5,weight_decay=0.0
