import torch
import os
import json
import time
from utils import utils
from utils import option
from data import dataset
from model.htr_bimamba_hybrid import create_model
import platform

import difflib

def highlight_diff(gt, pred):
    s = difflib.SequenceMatcher(None, gt, pred)
    out = []
    for tag, i1, i2, j1, j2 in s.get_opcodes():
        if tag == "equal":
            out.append(pred[j1:j2])
        elif tag in ("replace", "insert", "delete"):
            out.append(f"[{pred[j1:j2]}]")
    return "".join(out)


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

    logger.info("Loading training dataset for character set...")
    train_dataset = dataset.myLoadDS(args.train_data_list, args.data_path, args.img_size)
    logger.info("Loading test dataset...")
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

    img_names = []
    with open(args.test_data_list, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            img_path = line.split()[0]
            img_name = os.path.basename(img_path)
            img_names.append(img_name)

    assert len(img_names) == len(test_dataset), \
        f"Mismatch between the number of image names ({len(img_names)}) and the test dataset size ({len(test_dataset)})"

    logger.info("Starting inference...")
    all_preds = []
    all_gts = []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device, non_blocking=True)
            preds = model(imgs)                     # [B, T, C]
            preds = preds.permute(1, 0, 2).log_softmax(2)   # [T, B, C]

            _, preds_index = preds.max(2)           # [T, B]
            preds_size = torch.IntTensor([preds.size(0)] * imgs.size(0))
            preds_index = preds_index.transpose(1, 0).contiguous().view(-1)
            preds_str = converter.decode(preds_index.data, preds_size.data)

            all_preds.extend(preds_str)
            all_gts.extend(labels)

    log_file = os.path.join(args.save_dir, 'predictions.txt')
    with open(log_file, 'w', encoding='utf-8') as f:
        for img_name, gt, pred in zip(img_names, all_gts, all_preds):
            if gt == pred:
                f.write(f"{img_name}:GT:{gt} → Pred: {pred}\n")
            else:
                pred_marked = highlight_diff(gt, pred)
                f.write(f"{img_name}:GT:{gt} → Pred: {pred_marked} ⁜\n")

    logger.info(f"Predictions saved to {log_file}")
    logger.info(f"Total samples: {len(img_names)}")

    import editdistance
    tot_ed = 0
    tot_len = 0
    tot_ed_wer = 0
    tot_len_wer = 0
    for gt, pred in zip(all_gts, all_preds):
        # CER
        ed = editdistance.eval(pred, gt)
        tot_ed += ed
        tot_len += len(gt)
        # WER
        gt_wer = utils.format_string_for_wer(gt).split()
        pred_wer = utils.format_string_for_wer(pred).split()
        ed_wer = editdistance.eval(pred_wer, gt_wer)
        tot_ed_wer += ed_wer
        tot_len_wer += len(gt_wer)
    cer = tot_ed / tot_len if tot_len > 0 else 0.0
    wer = tot_ed_wer / tot_len_wer if tot_len_wer > 0 else 0.0
    logger.info(f"Overall CER: {cer:.4f}, WER: {wer:.4f}")

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    logger.info(f"System: {platform.system()} {platform.release()}  |  GPU: {gpu_name}")
    logger.info("Inference finished.")


if __name__ == "__main__":
    main()