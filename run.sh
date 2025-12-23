# what Anuj would run
# torchrun --standalone --nproc_per_node=8 nano_cli.py \
#   --num-devices 8 --batch-size 512 --num-iterations 5100 \
#   --scheduler cosine --warmup-steps 2000 --min-lr 0.0 \
#   --opt1 AdamW \
#   --opt1-kwargs lr=0.0036,betas=[0.9,0.95],eps=1e-8,weight_decay=0.0 \
#   --opt2 Muon \
#   --opt2-kwargs lr=0.02,momentum=0.95,nesterov=true,backend=newtonschulz5,backend_steps=5,weight_decay=0.0

# torchrun --standalone --nproc_per_node=8 nano_cli.py \
#   --num-devices 8 --batch-size 32 --device-batch-size 4 --num-iterations 20000 \
#   --scheduler cosine --warmup-steps 2000 --min-lr 0.0 \
#   --opt1 AdamW \
#   --opt1-kwargs lr=0.0036,betas=[0.9,0.95],eps=1e-8,weight_decay=0.0 \
#   --opt2 Muon \
#   --opt2-kwargs lr=0.02,momentum=0.95,nesterov=true,backend=newtonschulz5,backend_steps=5,weight_decay=0.1

# torchrun --standalone --nproc_per_node=8 nano_cli.py \
#   --num-devices 8 --batch-size 64 --device-batch-size 4 --num-iterations 20000 \
#   --scheduler cosine --warmup-steps 2000 --min-lr 0.0 \
#   --opt1 AdamW \
#   --opt1-kwargs lr=0.0036,betas=[0.9,0.95],eps=1e-8,weight_decay=0.0 \
#   --opt2 MuonScheduleFree \
#   --opt2-kwargs lr=0.1,momentum=0.95,backend=newtonschulz5,backend_steps=5,weight_decay=0.0

# torchrun --standalone --nproc_per_node=8 nano_cli.py \
#   --num-devices 8 --batch-size 512 --device-batch-size 64 --num-iterations 5100 \
#   --warmup-steps 2000 --min-lr 0.0 \
#   --opt1 AdamWScheduleFree \
#   --opt1-kwargs lr=0.0036,betas=[0.95,0.99],eps=1e-8,weight_decay=0.0 \
#   --opt2 MuonScheduleFree \
#   --opt2-kwargs lr=0.015,momentum=0.99,backend=newtonschulz5,backend_steps=5,weight_decay=0.0
  # momentum 0.99, lr =0.02 is the best so far, around 3.5
  # momentum 0.99, lr =0.025 val_loss:3.5013

  torchrun --standalone --nproc_per_node=8 nano_cli.py \
  --num-devices 8 --batch-size 512 --device-batch-size 64 --num-iterations 6200 \
  --warmup-steps 2000 --min-lr 0.0 \
  --opt1 AdamWScheduleFree \
  --opt1-kwargs lr=0.0036,betas=[0.95,0.99],eps=1e-8,weight_decay=0.0 \
  --opt2 NorMuonScheduleFree \
  --opt2-kwargs lr=0.03,betas=[0.95,0.95],eps=1e-8,eta_scale=0.2,backend=newtonschulz5,backend_steps=5,weight_decay=0.0



# lr=0.02,betas=[0.95,0.99] (--opt1-kwargs lr=0.0036,betas=[0.95,0.99]) step:5100/5100 val_loss:3.4066 train_time:2021838ms step_avg:397.22ms 
# lr=0.02,betas=[0.95,0.95] (normuon paper betas)  step:5100/5100 val_loss:3.4009 train_time:2034947ms step_avg:399.79ms
# lr=0.02,betas=[0.95,0.95] (--opt1-kwargs lr=0.0036,betas=[0.9,0.95]) val_loss: 10?... why suddenly?
# next try 0.00036 for step size (normuon+kellerjordan)
# step:6200/6200 opt1_lr=0.00360000 opt2_lr=0.00007195 opt2_ckp1=0.000205458 opt2_ycoef=-3.61156e-06train_loss:3.4109 train_time:2495903ms step_avg:403.22ms
# step:6200/6200 val_loss:3.5313 train_time:2495917ms step_avg:403.22ms

# 0.02 3.38xxx
# 0.023 step:6200/6200 val_loss:3.3921 train_time:2483136ms step_avg:401.15ms

# torchrun --standalone --nproc_per_node=8 nano_cli.py \
#   --num-devices 8 --batch-size 512 --device-batch-size 64 --num-iterations 5100 \
#   --warmup-steps 2000 --min-lr 0.0 \
#   --opt1 AdamWScheduleFree \
#   --opt1-kwargs lr=0.0036,betas=[0.95,0.99],eps=1e-8,weight_decay=0.0 \
#   --opt2 AdamWScheduleFree \
#   --opt2-kwargs lr=0.0036,betas=[0.95,0.99],eps=1e-8,weight_decay=0.0 