import json
import os

import fsspec
import hydra
import lightning as L
from lightning.fabric import Fabric
import omegaconf
import rich.syntax
import rich.tree
import torch
from torch.utils.data.distributed import DistributedSampler
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from tqdm import tqdm, trange

import algo
import dataloader
import utils

omegaconf.OmegaConf.register_new_resolver(
  'cwd', os.getcwd)
omegaconf.OmegaConf.register_new_resolver(
  'device_count', torch.cuda.device_count)
omegaconf.OmegaConf.register_new_resolver(
  'eval', eval)
omegaconf.OmegaConf.register_new_resolver(
  'div_up', lambda x, y: (x + y - 1) // y)


def _load_from_checkpoint(diffusion_model, config, tokenizer):
  if 'hf' in config.algo.backbone:
    return diffusion_model(
      config, tokenizer=tokenizer).to('cuda')
  
  return diffusion_model.load_from_checkpoint(
    config.eval.checkpoint_path,
    tokenizer=tokenizer,
    config=config)


@L.pytorch.utilities.rank_zero_only
def _print_config(
  config: omegaconf.DictConfig,
  resolve: bool = True,
  save_cfg: bool = True) -> None:
  """Prints content of DictConfig using Rich library and its tree structure.
  
  Args:
    config (DictConfig): Configuration composed by Hydra.
    resolve (bool): Whether to resolve reference fields of DictConfig.
    save_cfg (bool): Whether to save the configuration tree to a file.
  """

  style = 'dim'
  tree = rich.tree.Tree('CONFIG', style=style, guide_style=style)

  fields = config.keys()
  for field in fields:
    branch = tree.add(field, style=style, guide_style=style)

    config_section = config.get(field)
    branch_content = str(config_section)
    if isinstance(config_section, omegaconf.DictConfig):
      branch_content = omegaconf.OmegaConf.to_yaml(
        config_section, resolve=resolve)

    branch.add(rich.syntax.Syntax(branch_content, 'yaml'))
  rich.print(tree)
  if save_cfg:
    with fsspec.open(
      '{}/config_tree.txt'.format(
        config.checkpointing.save_dir), 'w') as fp:
      rich.print(tree, file=fp)


@L.pytorch.utilities.rank_zero_only
def _print_batch(config, train_ds, valid_ds, tokenizer, k=64):
  for dl_type, dl in [
    ('train', train_ds), ('valid', valid_ds)]:
    print(f'Printing {dl_type} dataloader batch.')
    batch = next(iter(dl))
    print('Batch input_ids.shape', batch['input_ids'].shape)
    if config.data.modality == 'text':
      first = batch['input_ids'][0, :k]
      last = batch['input_ids'][0, -k:]
      print(f'First {k} tokens:', tokenizer.decode(first))
      print('ids:', first)
      print(f'Last {k} tokens:', tokenizer.decode(last))
      print('ids:', last)



# def _generate_samples(diffusion_model, config, logger,
#                       tokenizer):
#     logger.info('Starting Sample Eval.')
#     model = _load_from_checkpoint(
#         diffusion_model=diffusion_model,
#         config=config,
#         tokenizer=tokenizer)
#     model.metrics.gen_ppl.reset()
#     model.metrics.sample_entropy.reset()
#     if config.eval.disable_ema:
#         logger.info('Disabling EMA.')
#         model.ema = None
#     stride_length = config.sampling.stride_length
#     num_strides = config.sampling.num_strides
#     all_samples = []

  
#     eps = 1e-5
#     steps = 1024
#     i = 1023
#     timesteps = model._get_sampling_time_profile(eps, steps)

#     x = torch.randint(1, 50257, (2, 1024))
    
#     t = timesteps[i] * torch.ones(x.shape[0], 1, device=model.device)
#     _, alpha_t = model.noise(t)
#     sigma = model._sigma_from_alphat(alpha_t)
    
#     log_x0_pred = model.forward(x, sigma)
#     p_x0 = log_x0_pred.exp()
    
#     model.tokenizer

import torch
from pathlib import Path


def _generate_samples(diffusion_model, config, logger, tokenizer):
    logger.info('Starting Sample Eval.')
    model = _load_from_checkpoint(
        diffusion_model=diffusion_model,
        config=config,
        tokenizer=tokenizer)
    model.metrics.gen_ppl.reset()
    model.metrics.sample_entropy.reset()
    if config.eval.disable_ema:
        logger.info('Disabling EMA.')
        model.ema = None
    text = Path("/home/nafez/scratch/duo/text_clean.txt").read_text(encoding="utf-8")
    token_ids = model.tokenizer.encode(text)
    chunk_size = 1024
    vocab_size = model.tokenizer.vocab_size
    n_chunks = len(token_ids) // chunk_size
    logger.info(
        f"Loaded {len(token_ids)} tokens "
        f"({n_chunks} chunks of {chunk_size})"
    )

    eps = 1e-5
    steps = 1024
    i = 1023
    timesteps = model._get_sampling_time_profile(eps, steps)

    total_correct = 0
    total_corrupted = 0
    total_copy_correct = 0
    total_clean_correct = 0
    total_num_clean = 0
    model.eval()

    with torch.no_grad():
        for chunk_idx in range(n_chunks):
            start = chunk_idx * chunk_size
            end = start + chunk_size

            clean_tokens = torch.tensor(
                token_ids[start:end],
                dtype=torch.long,
                device=model.device,
            ).unsqueeze(0)  # [1, 1024]
            x = clean_tokens.clone()
            # -------------------------------------------------
            # Corrupt 20% of positions
            # -------------------------------------------------
            corruption_mask = (
                torch.rand_like(x.float()) < 0.20
            )

            random_tokens = torch.randint(
                low=0,
                high=vocab_size,
                size=x.shape,
                device=x.device,
            )
            x[corruption_mask] = random_tokens[corruption_mask]
            num_corrupted = corruption_mask.sum().item()
            num_clean = (~corruption_mask).sum().item()
            if num_corrupted == 0:
                continue
            t = timesteps[i] * torch.ones(x.shape[0], 1, device=model.device)
            _, alpha_t = model.noise(t)
            sigma = model._sigma_from_alphat(alpha_t)
            sigma = model._process_sigma(sigma)

            with torch.amp.autocast('cuda', dtype=torch.float32):
                # pred_tokens = model.backbone(x=x, sigma=sigma, class_cond=None, weights=None, mask_embedding_blending=False, remove_self_attn=False).argmax(dim=-1)
                # pred_tokens = model.backbone(x=x, sigma=sigma, class_cond=None, weights=None, mask_embedding_blending=True, remove_self_attn=False).argmax(dim=-1)
                # pred_tokens = model.backbone(x=x, sigma=sigma, class_cond=None, weights=None, mask_embedding_blending=False, remove_self_attn=True).argmax(dim=-1)
                pred_tokens = model.backbone(x=x, sigma=sigma, class_cond=None, weights=None, mask_embedding_blending=True, remove_self_attn=True).argmax(dim=-1)

            
            correct = ((pred_tokens == clean_tokens) & corruption_mask).sum().item()
            copy_correct = ((x == pred_tokens) & corruption_mask).sum().item()
            clean_correct = ((clean_tokens == pred_tokens) & (~corruption_mask)).sum().item()
            

            '''
            corrupted_x = x[corruption_mask]
            corrupted_clean = clean_tokens[corruption_mask]
            corrupted_pred_tokens = pred_tokens[corruption_mask]
            print("\nFirst 20 corrupted positions:")
            for i in range(min(20, len(corrupted_x))):
                input_id = corrupted_x[i].item()
                target_id = corrupted_clean[i].item()
                pred_id = corrupted_pred_tokens[i].item()
                input_text = model.tokenizer.decode([input_id])
                target_text = model.tokenizer.decode([target_id])
                pred_text = model.tokenizer.decode([pred_id])
                print(
                    f"{i:02d}: "
                    f"input(x)={input_id:6d} ({repr(input_text)})  "
                    f"-> target={target_id:6d} ({repr(target_text)})  "
                    f"-> pred={pred_id:6d} ({repr(pred_text)})"
                )
            '''


            total_copy_correct += copy_correct
            total_correct += correct
            total_corrupted += num_corrupted

            total_clean_correct += clean_correct
            total_num_clean += num_clean

            if chunk_idx % 100 == 0:
                acc = total_correct / max(total_corrupted, 1)
                logger.info(
                    f"Chunk {chunk_idx}/{n_chunks} "
                    f"Acc={acc:.4f}"
                )

    final_acc = 100 * total_correct / max(total_corrupted, 1)
    model_acc = 100 * total_correct / total_corrupted
    copy_acc = 100 * total_copy_correct / total_corrupted

    clean_acc = 100 * total_clean_correct / total_num_clean
    overall_acc = 100 * (total_correct+total_clean_correct) / (total_corrupted+total_num_clean)

    


    logger.info("===============Final Results===============")
    logger.info(
        f"Denoising Accuracy (Prediction == Ground Truth on Corrupted Tokens): "
        f"{model_acc:.6f} ({total_correct}/{total_corrupted})"
    )

    logger.info(
        f"Copying Rate (Prediction == Corrupted Input on Corrupted Tokens): "
        f"{copy_acc:.6f} ({total_copy_correct}/{total_corrupted})"
    )
    logger.info(
        f"Clean Acc (Prediction == Input on Clean Tokens): "
        f"{clean_acc:.6f} ({total_clean_correct}/{total_num_clean})"
    )

    logger.info(
            f"Overall Acc (Prediction == Input): "
            f"{overall_acc:.6f} ({ (total_correct+total_clean_correct)}/{(total_corrupted+total_num_clean)})"
        )

  
    return final_acc


@hydra.main(version_base=None, config_path='configs',
            config_name='config')
def main(config):
  """Main entry point for training."""
  L.seed_everything(config.seed)
  _print_config(config, resolve=True, save_cfg=True)
  
  logger = utils.get_logger(__name__)
  tokenizer = dataloader.get_tokenizer(config)
  if config.algo.name == 'ar':
    diffusion_model = algo.AR
  elif config.algo.name == 'mdlm':
    diffusion_model = algo.MDLM
  elif config.algo.name == 'duo_base':
    diffusion_model = algo.DUO_BASE
  elif config.algo.name == 'd3pm':
    diffusion_model = algo.D3PMAbsorb
  elif config.algo.name == 'sedd':
    diffusion_model = algo.SEDDAbsorb
  elif config.algo.name == 'duo':
    diffusion_model = algo.DUO
  elif config.algo.name == 'distillation':
    diffusion_model = algo.Distillation
  elif config.algo.name == 'ot-finetune':
    diffusion_model = algo.OptimalTransportFinetune
  else:
    raise ValueError(
      f'Invalid algorithm name: {config.algo.name}')
  kwargs = {'diffusion_model': diffusion_model,
            'config': config,
            'tokenizer': tokenizer,
            'logger': logger}
  if config.mode == 'sample_eval':
    _generate_samples(**kwargs)
  

if __name__ == '__main__':
  main()