# Only do this step once
# conda create -n stripedhyena python=3.10
conda activate stripedhyena

pip install torch
pip install --no-build-isolation -r requirements.txt
# these are the model weights; download them directly instead of using
# the HF model implementation because it uses deprecated flash attention code
wget -c https://huggingface.co/togethercomputer/StripedHyena-Hessian-7B/resolve/main/pytorch-model.bin

# RUN THIS TO TEST (from README with ckpt path pointing to pytorch-model.bin)
# NOTE: don't use generate_transformers.py because it uses deprecated flash 
# attention code
# python generate.py --config_path ./configs/7b-sh-32k-v1.yml \
# --checkpoint_path pytorch-model.bin --cached_generation \
# --prompt_file ./test_prompt.txt

# If you want to look at GPU memory usage, you can use the following command:
# watch -n 1 nvidia-smi

# Example output:
# (sh-forked) kivelsons@thorshammer:~/bio-sae/stripedhyena$ python3 generate.py --config_path ./configs/7b-sh-32k-v1.yml --checkpoint_path pytorch-model.bin --cached_generation --prompt_file ./test_prompt.txt
# Loaded config: {'model_name': 'sh-7b-32k-v1', 'vocab_size': 32000, 'hidden_size': 4096, 'num_filters': 4096, 'attn_layer_idxs': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31], 'hyena_layer_idxs': [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32], 'num_layers': 32, 'short_filter_length': 3, 'num_attention_heads': 32, 'short_filter_bias': True, 'mlp_init_method': 'torch.nn.init.zeros_', 'mlp_output_init_method': 'torch.nn.init.zeros_', 'eps': 1e-05, 'state_size': 2, 'rotary_emb_base': 500000, 'make_vocab_size_divisible_by': 8, 'log_intermediate_values': False, 'proj_groups': 4, 'hyena_filter_groups': 1, 'column_split_hyena': True, 'column_split': False, 'model_parallel_size': 1, 'pipe_parallel_size': 1, 'tie_embeddings': False, 'inner_mlp_size': 14336, 'mha_out_proj_bias': False, 'qkv_proj_bias': False, 'max_seqlen': 32768, 'max_batch_size': 1, 'final_norm': True, 'use_flash_attn': True, 'use_flash_rmsnorm': True, 'use_flash_depthwise': False, 'use_flashfft': False, 'use_laughing_hyena': False, 'inference_mode': False, 'tokenizer_type': 'HFAutoTokenizer', 'vocab_file': 'tokenizer/tokenizer.json', 'prefill_style': 'fft'}
# /data1/home/kivelsons/miniconda3/envs/sh-forked/lib/python3.10/site-packages/flash_attn/ops/triton/layer_norm.py:985: FutureWarning: `torch.cuda.amp.custom_fwd(args...)` is deprecated. Please use `torch.amp.custom_fwd(args..., device_type='cuda')` instead.
#   def forward(
# /data1/home/kivelsons/miniconda3/envs/sh-forked/lib/python3.10/site-packages/flash_attn/ops/triton/layer_norm.py:1044: FutureWarning: `torch.cuda.amp.custom_bwd(args...)` is deprecated. Please use `torch.amp.custom_bwd(args..., device_type='cuda')` instead.
#   def backward(ctx, dout, *args):
# Loading state dict...

# /data1/home/kivelsons/bio-sae/stripedhyena/generate.py:42: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
#   m.load_state_dict(torch.load(args.checkpoint_path, map_location=device), strict=False)
# Prompt: The four species of hyenas are

# Memory after tokenization: 15.303858176 GB
# Starting generation...
# Prompt: The four species of hyenas are
# the stri ped hy ena ( H ya ena h ya ena ), the brown hy ena ( Par ah ya ena br un nea ), the spotted hy ena ( C ro cut a cro cut a ), and the a ard w olf ( Pro t eles crist ata ). 
 
#  The stri ped hy ena is the most widespread species , occurring in Africa , the Middle East , and Asia . The brown hy ena is found in southern 
# Input: The four species of hyenas are, Output: ['the striped hyena (Hyaena hyaena), the brown hyena (Parahyaena brunnea), the spotted hyena (Crocuta crocuta), and the aardwolf (Proteles cristata).\n\nThe striped hyena is the most widespread species, occurring in Africa, the Middle East, and Asia. The brown hyena is found in southern']
# Memory after generation: 17.595986944 GB
