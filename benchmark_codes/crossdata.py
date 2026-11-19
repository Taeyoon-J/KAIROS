#!/usr/bin/env python3
"""Zero-shot: does the learned head generalise across datasets better than spike clustering?

The head was not added for CLEVR score -- it was added for dataset generality, and the peer
measured it as slightly better on their own small test. Every experiment I ran was on CLEVR,
so none of them addressed that.

There is a concrete reason to expect spike clustering to transfer badly: it has hand-set
thresholds, and I tuned them ON CLEVR hours ago -- min_group_size 8 vs 32 vs 64 moved CLEVR
FG-ARI by 0.055. A learned head has no such threshold. If A and B separate differently on
each dataset, that fragility is measured rather than argued.

One CLEVR-trained core, frozen, evaluated zero-shot on Tetrominoes and Multi-dSprites:
  A  spike clustering, min_group_size=8   (tuned on CLEVR validation)
  B  spike clustering, min_group_size=32  (the setting in use until today)
  C  the CLEVR-trained context head
  D  all-ones, no foreground detection    (reference ceiling for FG-ARI)

C and D designate no background, so `prediction != 0` is the whole image and their
Foreground IoU is the ground-truth foreground fraction, not a detection score. Compare C
against A and B on FG-ARI and Object IoU only.
"""
import argparse, copy, json, sys, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ap = argparse.ArgumentParser()
ap.add_argument('--code', required=True); ap.add_argument('--config', required=True)
ap.add_argument('--run', required=True); ap.add_argument('--fit', required=True)
ap.add_argument('--dataset', required=True); ap.add_argument('--count', type=int, default=320)
ap.add_argument('--out', required=True)
a = ap.parse_args()
CODE = Path(a.code); sys.path.insert(0, str(CODE/'vendor')); sys.path.insert(0, str(CODE))
import portable_runtime as pr
config = pr.configure(a.config); output = pr.connect_inputs(config, a.run)
from full_benchmark.data import Dataset
from full_benchmark.model import FreshModel
from mod_benchmark import common as c
from readout_benchmark.head import activity_features
from neutral_readout.head import NeutralRegionHead, candidates
from snn_kuramoto_bidirectional.spike_classifier import spike_synchrony_components
from snn_kuramoto_bidirectional.evaluation import spatial_components_to_patch_labels
from full_benchmark.fast_partition import foreground_induced_merge

GRID, NODES = 64, 4096

class ContextAdapter(nn.Module):
    def __init__(self, seed, dilation=1):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(262000+seed)
            self.net = nn.Sequential(nn.Conv2d(8,16,3,padding=dilation,dilation=dilation), nn.SiLU(),
                                     nn.Conv2d(16,32,3,padding=dilation,dilation=dilation,bias=False))
            nn.init.zeros_(self.net[-1].weight)
    def forward(self, g):
        return self.net(g.reshape(GRID,GRID,8).permute(2,0,1)[None])[0].permute(1,2,0).reshape(NODES,32)

class R(nn.Module):
    def __init__(self, seed, radius):
        super().__init__(); self.radius = radius
        base = NeutralRegionHead(seed=259000+seed); self.head = base
        self.content_encoder = copy.deepcopy(base.embedding)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(259100+seed)
            self.decoder = nn.Sequential(nn.Linear(34,64), nn.SiLU(), nn.Linear(64,64), nn.SiLU(),
                                         nn.Linear(64,3), nn.Sigmoid())
        self.fine = ContextAdapter(seed)
        axis = (torch.arange(GRID, dtype=torch.float32)+.5)/(GRID/2)-1
        yy, xx = torch.meshgrid(axis, axis, indexing='ij')
        self.register_buffer('xy', torch.stack([xx.flatten(), yy.flatten()], -1))
    @torch.no_grad()
    def predict(self, f, reg, g):
        masks, prior = candidates(f, reg)
        z = F.normalize(self.head.embedding(f) + self.fine(g), dim=-1)
        proto = F.normalize(masks.float() @ z / masks.sum(1,keepdim=True).clamp_min(1), dim=-1)
        sim = proto @ z.T; r = self.radius
        allowed = F.max_pool2d(masks.float().reshape(-1,1,GRID,GRID), 2*r+1, stride=1, padding=r).flatten(1).bool()
        w = allowed.float(); mean = (sim*w).sum(0)/w.sum(0).clamp_min(1)
        std = (((sim-mean).square()*w).sum(0)/w.sum(0).clamp_min(1)).clamp_min(.0025).sqrt()
        return ((prior + 2.*((sim-mean)/std).tanh()).masked_fill(~allowed,-1e4).argmax(0)+1
                ).reshape(GRID,GRID).cpu().numpy().astype(np.int64)

job = Path(a.run)/'jobs/model253/clevr/seed0'
resume = torch.load(job/'resume.pt', map_location='cuda:0', weights_only=True)
initial = torch.load(job/'initial.pt', map_location='cpu', weights_only=True)
model = FreshModel(253, 0, normalization=initial['normalization']).to('cuda:0')
model.load_state_dict(resume['model']); model.eval().requires_grad_(False)
gain = float(model.wrapped.core.graph_generator.log_coupling_gain.exp().detach().cpu())

sel = json.loads((Path(a.fit)/'selected.json').read_text())
blob = torch.load(Path(a.fit)/f"{sel['arm']}_best.pt", map_location='cuda:0', weights_only=True)
head = R(0, 1).cuda().eval(); head.load_state_dict(blob['model'])
print(f"head: arm {sel['arm']}, epoch {blob['epoch']} (trained on CLEVR)", flush=True)

dataset = Dataset(a.dataset, output)
ids = dataset.test[:a.count]
print(f"{a.dataset}: {len(ids)} TEST images, zero-shot from a CLEVR-trained core", flush=True)
P = {k: [] for k in ('A_min8','B_min32','C_head','D_all_ones')}
fgf = {'A_min8': [], 'B_min32': []}
t0 = time.time()
for n, image_id in enumerate(ids):
    rgb = dataset.rgb(np.array([image_id]), 'test')[0]
    with torch.no_grad():
        o = model(torch.from_numpy(rgb[None].copy()).cuda(), inference_seed=193000+n, live_tail_steps=0)
    sp = o['trace']['spikes'].detach().cpu(); cs = o['trace']['component_spikes'].detach().cpu()
    q = o['q'][0].detach().cpu().numpy().astype(np.float64)
    adj = o['adjacency'][0].detach().cpu().numpy().astype(np.float64)
    allones = foreground_induced_merge(q, adj, gain, np.ones((GRID,GRID),dtype=np.int64), grid_shape=(GRID,GRID))['labels']
    P['D_all_ones'].append(allones)
    for name, mg in (('A_min8',8), ('B_min32',32)):
        g = spike_synchrony_components(sp, foreground_threshold=.5, synchrony_threshold=.5,
            min_group_size=mg, settle=512, components=cs, background='largest_component',
            affinity_mode='spike', spatial_grid_size=GRID)
        sup = spatial_components_to_patch_labels(g, GRID, device='cpu').numpy()[0].astype(np.int64)
        fgf[name].append(float((sup>0).mean()))
        P[name].append(foreground_induced_merge(q, adj, gain, sup, grid_shape=(GRID,GRID))['labels'])
    # the head consumes the same whole-image partition that regions_from_output builds
    feats = activity_features(o['trace'])[0]
    gm = o['gamma'][0]
    gm = gm.T.contiguous() if gm.shape[0] == 8 else gm.contiguous()
    P['C_head'].append(head.predict(feats, allones, gm))
    del o, sp, cs
    if (n+1) % 20 == 0:
        el = time.time()-t0
        print(f"  {n+1}/{len(ids)}  {el/(n+1):.1f} s/image  eta {(len(ids)-n-1)*el/(n+1)/60:.0f} min", flush=True)

metric = c.benchmark_module('evaluation')
target = torch.from_numpy(dataset.labels(ids, 'test'))
KEYS = ('fg_ari','foreground_iou','matched_object_iou')
truth_fg = float((target.numpy() > 0).mean())
print(f"\n{a.dataset}: ground-truth foreground fraction {truth_fg:.4f}\n", flush=True)
table = {'_dataset': a.dataset, '_count': len(ids), '_truth_foreground_fraction': truth_fg,
         '_note': 'zero-shot from a CLEVR-trained core; C and D designate no background so their foreground_iou is the truth fraction, not a detection score'}
for name in ('A_min8','B_min32','C_head','D_all_ones'):
    s = metric.evaluate_patch_masks(torch.from_numpy(np.stack(P[name]).astype(np.int64)), target)
    row = {k: float(s['mean'][k]) for k in KEYS}
    if name in fgf: row['pred_fg'] = float(np.mean(fgf[name]))
    table[name] = row
    print(f"{name:12s} fg_ari {row['fg_ari']:.4f}  fg_iou {row['foreground_iou']:.4f}  "
          f"obj_iou {row['matched_object_iou']:.4f}" + (f"  pred_fg {row['pred_fg']:.4f}" if 'pred_fg' in row else ""), flush=True)
Path(a.out).write_text(json.dumps(table, indent=1)+'\n')
print("DONE ->", a.out, flush=True)
