"""Zero-shot evaluation of Grounding DINO on LVIS (v1 val / minival).

Khác với COCO (80 class nhét trong 1 prompt), LVIS có 1203 class nên phải:
  1. Chia category thành nhiều "chunk" sao cho caption tokenize <= 256 token.
  2. Chạy model 1 lần / chunk / ảnh, gộp detection, giữ top-300 / ảnh.
  3. Tính AP / APr / APc / APf bằng lvis-api.

Tái sử dụng đúng các helper đã chạy được trong test_ap_on_coco.py:
  build_captions_and_token_span, create_positive_map_from_span, box_ops, get_tokenlizer.
"""
import argparse
import os
import time
import json

import torch
from PIL import Image
from torch.utils.data import DataLoader

from groundingdino.models import build_model
import groundingdino.datasets.transforms as T
from groundingdino.util import box_ops, get_tokenlizer
from groundingdino.util.misc import clean_state_dict, collate_fn
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.vl_utils import create_positive_map_from_span

from lvis import LVIS, LVISEval, LVISResults


def load_model(model_config_path, model_checkpoint_path, device="cuda"):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    model.eval()
    return model


class LvisDetection(torch.utils.data.Dataset):
    """Trả về (image_tensor, target) — target chỉ cần image_id + orig_size để eval."""

    def __init__(self, image_root, lvis_api, transforms, max_images=None):
        self.image_root = image_root
        self.lvis = lvis_api
        ids = sorted(lvis_api.get_img_ids())
        if max_images is not None:
            ids = ids[:max_images]
        self.ids = ids
        self._transforms = transforms

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        info = self.lvis.load_imgs([img_id])[0]
        # LVIS v1: coco_url = http://images.cocodataset.org/<split>/<file>.jpg
        url = info["coco_url"]
        split = url.split("/")[-2]      # train2017 / val2017
        fname = url.split("/")[-1]
        path = os.path.join(self.image_root, split, fname)
        img = Image.open(path).convert("RGB")
        w, h = img.size
        target = {
            "image_id": img_id,
            "orig_size": torch.as_tensor([int(h), int(w)]),
            "boxes": torch.zeros((0, 4)),  # rỗng, chỉ để transform không lỗi
        }
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target


def build_chunks(cat_names, cat_ids, tokenizer, max_tokens=250):
    """Greedy: gộp category tới khi caption tokenize gần chạm max_tokens thì sang chunk mới."""
    chunks = []
    cur_names, cur_ids = [], []
    for name, cid in zip(cat_names, cat_ids):
        trial = cur_names + [name]
        cap = " . ".join(trial) + " ."
        if len(tokenizer(cap)["input_ids"]) > max_tokens and cur_names:
            chunks.append((cur_names, cur_ids))
            cur_names, cur_ids = [name], [cid]
        else:
            cur_names, cur_ids = trial, cur_ids + [cid]
    if cur_names:
        chunks.append((cur_names, cur_ids))
    return chunks


def build_caption_and_spans(names):
    """Tu build caption + char-span theo INDEX, KHONG dung build_captions_and_token_span
    de tranh KeyError (do lowercase) va non-determinism (do random.choice khi ten co '/').
    Tra ve caption va spans[i] = list cac [beg, end] cua category thu i (khop theo thu tu)."""
    caption = ""
    spans = []
    for name in names:
        class_name = name.replace("/", " ").strip().lower()
        toks = []
        for sub in class_name.split(" "):
            sub = sub.strip()
            if len(sub) == 0:
                continue
            if len(caption) > 0:
                caption = caption + " "
            start = len(caption)
            end = start + len(sub)
            toks.append([start, end])
            caption = caption + sub
        caption = caption + " ."
        spans.append(toks)
    return caption, spans


def build_chunk_posmaps(chunks, tokenizer):
    """Tiền tính (caption, positive_map, lvis_cat_ids) cho từng chunk — không phụ thuộc ảnh."""
    out = []
    for names, ids in chunks:
        caption, spans = build_caption_and_spans(names)
        pm = create_positive_map_from_span(tokenizer(caption), spans)  # (len(names), 256)
        out.append((caption, pm, ids))
    return out


@torch.no_grad()
def decode_chunk(outputs, target_sizes, positive_map, chunk_ids, num_select=300):
    out_logits = outputs["pred_logits"]            # (bs, nq, 256)
    out_bbox = outputs["pred_boxes"]               # (bs, nq, 4)
    pm = positive_map.to(out_logits.device)
    prob = out_logits.sigmoid() @ pm.T             # (bs, nq, n_chunk)
    bs, nq, ncat = prob.shape
    k = min(num_select, nq * ncat)
    topv, topi = torch.topk(prob.reshape(bs, -1), k, dim=1)
    box_idx = topi // ncat
    cat_idx = topi % ncat
    boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
    boxes = torch.gather(boxes, 1, box_idx.unsqueeze(-1).repeat(1, 1, 4))
    img_h, img_w = target_sizes.unbind(1)
    scale = torch.stack([img_w, img_h, img_w, img_h], dim=1)
    boxes = boxes * scale[:, None, :]
    chunk_ids_t = torch.as_tensor(chunk_ids, device=out_logits.device)
    res = []
    for b in range(bs):
        res.append({
            "scores": topv[b].detach().cpu(),
            "labels": chunk_ids_t[cat_idx[b]].detach().cpu(),
            "boxes": boxes[b].detach().cpu(),
        })
    return res


def main(args):
    device = args.device
    cfg = SLConfig.fromfile(args.config_file)

    model = load_model(args.config_file, args.checkpoint_path, device).to(device).eval()
    tokenizer = get_tokenlizer.get_tokenlizer(cfg.text_encoder_type)

    lvis_api = LVIS(args.anno_path)

    # danh sách category theo thứ tự id tăng dần
    cat_ids = sorted(lvis_api.cats.keys())
    cat_names = [lvis_api.cats[c]["name"].replace("_", " ") for c in cat_ids]
    print(f"#categories = {len(cat_ids)}")

    chunks = build_chunks(cat_names, cat_ids, tokenizer, max_tokens=args.max_tokens)
    chunk_data = build_chunk_posmaps(chunks, tokenizer)
    print(f"#chunks = {len(chunk_data)} (mỗi ảnh chạy model {len(chunk_data)} lần)")

    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    dataset = LvisDetection(args.image_root, lvis_api, transform, max_images=args.max_images)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=collate_fn)
    print(f"#images = {len(dataset)}")

    results = []
    start = time.time()
    for it, (images, targets) in enumerate(loader):
        images = images.tensors.to(device)
        bs = images.shape[0]
        orig = torch.stack([t["orig_size"] for t in targets]).to(device)
        img_ids = [int(t["image_id"]) for t in targets]

        accum = [{"scores": [], "labels": [], "boxes": []} for _ in range(bs)]
        for caption, pm, ids in chunk_data:
            outputs = model(images, captions=[caption] * bs)
            dec = decode_chunk(outputs, orig, pm, ids, args.num_select)
            for b in range(bs):
                accum[b]["scores"].append(dec[b]["scores"])
                accum[b]["labels"].append(dec[b]["labels"])
                accum[b]["boxes"].append(dec[b]["boxes"])

        for b in range(bs):
            scores = torch.cat(accum[b]["scores"])
            labels = torch.cat(accum[b]["labels"])
            boxes = torch.cat(accum[b]["boxes"], dim=0)
            k = min(300, scores.shape[0])
            tv, ti = torch.topk(scores, k)
            for s, idx in zip(tv.tolist(), ti.tolist()):
                x1, y1, x2, y2 = boxes[idx].tolist()
                results.append({
                    "image_id": img_ids[b],
                    "category_id": int(labels[idx].item()),
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": float(s),
                })

        if (it + 1) % 20 == 0:
            done = (it + 1) * bs
            el = time.time() - start
            eta = el / done * (len(dataset) - done)
            print(f"{done}/{len(dataset)} imgs | {len(results)} dets | {el:.0f}s | ETA {eta:.0f}s")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f)
        print(f"Saved detections -> {args.out}")

    # ----- LVIS eval -----
    lvis_dt = LVISResults(lvis_api, results, max_dets=300)
    lvis_eval = LVISEval(lvis_api, lvis_dt, "bbox")
    lvis_eval.params.img_ids = dataset.ids   # chỉ chấm trên các ảnh đã chạy
    lvis_eval.run()
    lvis_eval.print_results()


if __name__ == "__main__":
    p = argparse.ArgumentParser("Grounding DINO eval on LVIS")
    p.add_argument("--config_file", "-c", type=str, required=True)
    p.add_argument("--checkpoint_path", "-p", type=str, required=True)
    p.add_argument("--anno_path", type=str, required=True, help="LVIS json (val hoặc minival)")
    p.add_argument("--image_root", type=str, required=True,
                   help="thư mục chứa train2017/ và val2017/ (vd .../coco2017)")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--num_select", type=int, default=300)
    p.add_argument("--max_tokens", type=int, default=250)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_images", type=int, default=None,
                   help="chỉ chạy N ảnh đầu (để test nhanh)")
    p.add_argument("--out", type=str, default=None, help="lưu detections ra json")
    main(p.parse_args())
