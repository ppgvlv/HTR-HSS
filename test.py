import torch
import os
import json
import time
import valid
from utils import utils
from utils import option
from data import dataset
from model.htr_bimamba_hybrid import create_model
from collections import OrderedDict
import platform


def main():
    args = option.get_args_parser()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    args.save_dir = os.path.join(args.out_dir, args.exp_name)
    os.makedirs(args.save_dir, exist_ok=True)
    logger = utils.get_logger(args.save_dir)
    logger.info(json.dumps(vars(args), indent=4, sort_keys=True))
    
    model = create_model(
        nb_cls=args.nb_cls,
        img_size=args.img_size[::-1],
        embed_dim=args.embed_dim,
        depth=args.encoder_depth,
        mlp_ratio=args.mlp_ratio,
        drop_path_rate=args.drop_path_rate,
        num_levels=args.num_levels,
        channel_multiplier=args.channel_multiplier,
        td_stride=args.td_stride,
        attn_every=args.attn_every,
        attn_heads=args.attn_heads,
        attn_window=args.attn_window,
        use_asymmetric=not args.no_aniso,
        use_csp=not args.no_csp,
        enable_mamba=not args.no_mamba,
        enable_attn=not args.no_attn,
    )
    model = model.to(device)
    model.eval()

    pth_path = os.path.join(args.save_dir, 'best_CER.pth')
    logger.info(f"loading HWR checkpoint from {pth_path}")
    ckpt = torch.load(pth_path, map_location="cpu")
    state_dict = ckpt.get('state_dict_ema', ckpt)
    if list(state_dict.keys())[0].startswith("module."):
        logger.info("Detected 'module.' prefix in checkpoint keys, removing...")
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    model_size = os.path.getsize(pth_path) / (1024 ** 2)
    logger.info(f"Model Parameters: {total_params:.2f} M")
    logger.info(f"Model File Size: {model_size:.2f} MB")

    logger.info("Loading test loader...")
    train_dataset = dataset.myLoadDS(args.train_data_list, args.data_path, args.img_size)
    test_dataset = dataset.myLoadDS(
        args.test_data_list, args.data_path, args.img_size, ralph=train_dataset.ralph
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.val_bs,
        shuffle=False,
        pin_memory=True,
        num_workers=max(4, args.num_workers),
    )

    converter = utils.CTCLabelConverter(train_dataset.ralph.values())
    criterion = torch.nn.CTCLoss(reduction='none', zero_infinity=True).to(device)

    total_time = 0.0
    total_samples = 0
    batch_times = []

    logger.info("Starting inference timing...")
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device, non_blocking=True)
            torch.cuda.synchronize()
            t0 = time.time()
            _ = model(imgs)
            torch.cuda.synchronize()
            t1 = time.time()
            batch_times.append(t1 - t0)
            total_time += t1 - t0
            total_samples += imgs.size(0)

        val_loss, val_cer, val_wer, preds, gts = valid.validation(
            model, criterion, test_loader, converter
        )

    avg_batch_time = sum(batch_times) / len(batch_times)
    avg_time_per_sample = avg_batch_time / args.val_bs * 1000
    fps = args.val_bs / avg_batch_time
    throughput = total_samples / total_time

    logger.info("=" * 59)
    logger.info("📊 Final Test Results:")
    logger.info(f"Loss: {val_loss:.3f}")
    logger.info(f"CER: {val_cer:.4f}")
    logger.info(f"WER: {val_wer:.4f}")
    logger.info(f"Parameters: {total_params:.2f} M")
    logger.info(f"Model Size: {model_size:.2f} MB")
    logger.info(f"Avg Inference Time: {avg_time_per_sample:.2f} ms/sample")
    logger.info(f"FPS (batch={args.val_bs}): {fps:.2f}")
    logger.info(f"Throughput (batch={args.val_bs}): {throughput:.2f} images/sec")
    logger.info("=" * 59)

    if preds and gts:
        logger.info("Example predictions (GT → Pred):")
        for i in range(min(5, len(preds))):
            logger.info(f"[{i+1}] GT: {gts[i]}  →  Pred: {preds[i]}")

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    logger.info(f"System: {platform.system()} {platform.release()}  |  GPU: {gpu_name}")


if __name__ == "__main__":
    main()
