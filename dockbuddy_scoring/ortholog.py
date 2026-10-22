# -*- coding: utf-8 -*-
"""
scoring.ortholog —— OrthologTransformer 跨物种基因重设计适配器(2026-10-10)
================================================================================
定位: 把 Nat.Commun. 2026 的 OrthologTransformer(DOI 10.1038/s41467-026-69966-0)
封装为 pipeline 的「基因重设计」组件 —— 把某物种的编码 DNA 序列(CDS)改写成适配
目标宿主的版本(同义替换 + 保守非同义突变 + 小片段 indel), 返回重设计后的 DNA 与
翻译蛋白序列。

与其他打分模型同款接入口径:
  - 模块级**轻量导入**(延迟 import torch / Bio), 缺依赖时抛带部署指引的 RuntimeError;
  - 权重(aligned_model.pt / unaligned_model.pt, 各 ~435MB)不进代码库, 本地放
    deploy/models/ortholog/assets/, 部署时上传到实例 /opt/dockbuddy/models/ortholog/
    (见 deploy/models/ortholog/README.md 与 deploy/镜像安装正向更新计划.md);
  - 推理只需 CPU 亦可用(模型 1.14 亿参数, GPU 秒级 / CPU 分钟级), 自动按
    torch.cuda.is_available() 选 device。

原仓库 github.com/mana438/OrthologTransformer 的核心推理逻辑(run_inference.py +
model.py + read.py 的 CodonTransformer/Vocab/greedy 生成)已收敛到本模块, 不依赖
原仓库运行时; vocab_OMA.json 收敛为 ortholog_vocab.json(本包资源文件)。

使用:
  from scoring.ortholog import redesign
  out = redesign("ATGAACTTT...", source_species="IDESA", target_species="BACSU",
                 model_path="/path/to/aligned_model.pt")
  # -> {"predicted_dna": "...", "predicted_protein": "...", "source_protein": "..."}

诚实边界: 重设计不改蛋白功能(保守位点几乎不动, 见 dN/dS 分析), 它解决的是
「异源表达」而非「活性改造」; 输出是宿主适配的编码序列, 表达水平需湿实验验证。
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, Optional, Tuple

LOG = logging.getLogger("dockpipeline")
if not LOG.handlers:
    # 独立运行时(不经过 pipeline common 的 basicConfig)兜底, 保证进度日志可见。
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [ortholog] %(levelname)s: %(message)s",
                        datefmt="%H:%M:%S")

# 本包资源: vocab(单一事实源, 与官方 vocab_OMA.json 一致, 2206 token 含 2138 物种)
_VOCAB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "ortholog_vocab.json")

# 官方预训练超参(与 arg_parser.py 默认 + 成功复现所用 args_test.json 完全一致):
# d_model=512, nhead=16, encoder/decoder=20 层, dim_feedforward=1024, dropout=0.0。
# ⚠ dim_feedforward 是 1024 而非 2048, dropout 是 0.0 而非 0.1 —— 写错会导致
#    state_dict 尺寸不匹配(load_state_dict strict=True 直接抛错)。
_D_MODEL = 512
_NHEAD = 16
_N_ENC = 20
_N_DEC = 20
_DIM_FF = 1024
_DROPOUT = 0.0

# vocab 里密码子 token 占据索引 3..66(<pad>=0/<bos>=1/<eos>=2, gap '---'=67,
# 物种 ID 从 68 起)。解码 DNA 时只保留 3..66。
_CODON_LO, _CODON_HI = 3, 66

_ORTHOLOG_HINT = (
    "OrthologTransformer 跨物种基因重设计需要: torch>=2.0 + biopython>=1.85, "
    "以及预训练权重(aligned_model.pt / unaligned_model.pt, 各约 435MB)。"
    "权重不进代码库: 本地放 deploy/models/ortholog/assets/, 部署时上传到实例 "
    "/opt/dockbuddy/models/ortholog/(见 deploy/镜像安装正向更新计划.md), "
    "或经 model_path 显式指定。安装: pip install torch biopython numpy。"
)


# ---------------------------------------------------------------------------
# CodonTransformer(与官方 model.py 一致, 仅搬移)
# ---------------------------------------------------------------------------
def _codon_transformer_cls():
    import torch
    import torch.nn as nn

    class CodonTransformer(nn.Module):
        def __init__(self, vocab_size_target, vocab_size_input, d_model, nhead,
                     num_encoder_layers, num_decoder_layers, dim_feedforward, dropout):
            super().__init__()
            self.embedding_e = nn.Embedding(vocab_size_input, d_model)
            self.embedding_d = nn.Embedding(vocab_size_input, d_model)
            self.positional_encoding_e = nn.Embedding(1000, d_model)
            self.positional_encoding_d = nn.Embedding(1000, d_model)
            self.encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                dropout=dropout, batch_first=True, norm_first=True)
            self.encoder = nn.TransformerEncoder(self.encoder_layer, num_encoder_layers)
            self.decoder_layer = nn.TransformerDecoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                dropout=dropout, batch_first=True, norm_first=True)
            self.decoder = nn.TransformerDecoder(self.decoder_layer, num_decoder_layers)
            self.fc_out_1 = nn.Linear(d_model, d_model)
            self.fc_out_2 = nn.Linear(d_model, int(d_model / 2))
            self.fc_out_3 = nn.Linear(int(d_model / 2), vocab_size_target)
            self.act = nn.ReLU()
            self.d_model = d_model

        def forward(self, tgt, src, src_key_padding_mask=None, tgt_key_padding_mask=None):
            src = self.embedding_e(src)
            src = src + self.positional_encoding_e(
                torch.arange(src.size(dim=1)).unsqueeze(0).repeat(src.size(0), 1).to(src.device))
            memory = self.encoder(src, src_key_padding_mask=src_key_padding_mask)
            tgt = self.embedding_d(tgt)
            tgt = tgt + self.positional_encoding_d(
                torch.arange(tgt.size(dim=1)).unsqueeze(0).repeat(tgt.size(0), 1).to(tgt.device))
            tgt_mask = nn.Transformer().generate_square_subsequent_mask(tgt.size(1)).to(src.device)
            output = self.decoder(tgt, memory, tgt_mask=tgt_mask,
                                  tgt_key_padding_mask=tgt_key_padding_mask,
                                  memory_key_padding_mask=src_key_padding_mask)
            return self.fc_out_3(self.act(self.fc_out_2(self.act(self.fc_out_1(output)))))

    return CodonTransformer


# ---------------------------------------------------------------------------
# vocab 与 token <-> 序列 转换
# ---------------------------------------------------------------------------
def _load_vocab(vocab_path: str = "") -> Tuple[Dict[str, int], Dict[int, str]]:
    p = vocab_path or _VOCAB_PATH
    if not os.path.exists(p):
        raise RuntimeError(f"未找到 OrthologTransformer vocab: {p}")
    data = json.load(open(p, encoding="utf-8"))
    t2i = data["token_to_index"]
    i2t = {int(k): v for k, v in data.get("index_to_token", {}).items()}
    return t2i, i2t


def _dna_to_tokens(dna: str, species_idx: int, vocab: Dict[str, int]) -> list:
    """CDS -> token 列表 [species, <bos>, codon..., <eos>] (与官方 convert_to_codons 同口径)。"""
    dna = (dna or "").strip().upper()
    if not dna or len(dna) % 3 != 0 or len(dna) > 2100:
        raise ValueError(f"编码序列非法(需为 3 的倍数且 ≤2100 bp): 长度 {len(dna)}")
    if set(dna) - set("ATGC"):
        raise ValueError("编码序列含非 ATGC 字符")
    codons = [dna[i:i + 3] for i in range(0, len(dna), 3)]
    return ([species_idx, vocab["<bos>"]] +
            [vocab[c] for c in codons] + [vocab["<eos>"]])


def _tokens_to_dna(token_list: list, i2t: Dict[int, str]) -> str:
    """token 列表 -> DNA 字符串(只保留密码子 token 3..66)。

    剔除此三类非密码子 token:
      - special 0/1/2(<pad>/<bos>/<eos>);
      - gap '---'(67);
      - 物种 ID(≥68, 如 BACSU=916 —— 若不清除会混进输出 DNA, 导致翻译错位)。
    """
    return "".join(i2t.get(t, "") for t in token_list if _CODON_LO <= t <= _CODON_HI)


def _resolve_model(model_path: str) -> str:
    """定位预训练权重: 显式 model_path > DOCK_ORTHOLOG_MODEL > 镜像缺省目录。"""
    p = model_path or os.environ.get("DOCK_ORTHOLOG_MODEL", "")
    if p and os.path.isfile(p):
        return p
    cands = [
        os.path.join(os.environ.get("DOCK_MODELS_DIR", "/opt/dockbuddy/models"),
                     "ortholog", "aligned_model.pt"),
        os.path.join(os.path.expanduser("~"), ".config", "ortholog", "aligned_model.pt"),
    ]
    for c in cands:
        if os.path.isfile(c):
            return c
    raise RuntimeError(_ORTHOLOG_HINT + f"\n未找到权重(model_path={model_path!r})。")


# ---------------------------------------------------------------------------
# beam search(高 GC 序列 greedy 易退化: 不生成 EOS 跑满 max_len; 见官方 README)
# ---------------------------------------------------------------------------
def _beam_search(model, src, dec_ipt, eos, beam_size: int, max_len: int):
    """Vanilla beam search(单序列 batch=1), 返回最高分 beam 的 token 序列 [1, L]。

    greedy 对高 GC 源序列(如放线菌 ~70% GC)会退化为「一直生成、不产生 EOS」;
    beam 探索多条路径, 取累计 log-prob 最高者, 稳健得多。

    实现要点(性能): 每步把所有 alive beam 堆成 batch 做**一次** forward(而非逐 beam),
    快 ~beam_size 倍; topk 用 .tolist() 一次性拉回, 避免大量 .item() GPU→CPU 同步。
    """
    import torch
    import torch.nn.functional as F

    beams = [(dec_ipt, 0.0)]  # (seq[1,L], 累计 log-prob)
    for _ in range(max_len):
        alive = [(s, sc) for s, sc in beams if not bool((s[0, -1] == eos).item())]
        done = [(s, sc) for s, sc in beams if bool((s[0, -1] == eos).item())]
        if not alive:
            break
        batch = torch.cat([s for s, _ in alive], dim=0)  # [B, L] alive 等长, 直接堆
        logits = model(batch, src.expand(batch.size(0), -1))
        logp = F.log_softmax(logits[:, -1], dim=-1)  # [B, vocab]
        topk_v, topk_i = torch.topk(logp, beam_size, dim=-1)
        v_list, i_list = topk_v.tolist(), topk_i.tolist()
        cands = list(done)
        for bi, (s, sc) in enumerate(alive):
            for k in range(len(i_list[bi])):
                ns = torch.cat([s, torch.tensor([[i_list[bi][k]]], device=s.device)], dim=1)
                cands.append((ns, sc + v_list[bi][k]))
        cands.sort(key=lambda x: -x[1])
        beams = cands[:beam_size]
    return beams[0][0]


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
def redesign(source_dna: str, source_species: str, target_species: str,
             model_path: str = "", vocab_path: str = "",
             max_len: int = 700, device: str = "auto",
             beam_size: int = 1, fixed_aa_len: int = 0, **cfg) -> Dict[str, str]:
    """跨物种基因重设计: 把 source 物种的 CDS 改写为 target 宿主适配版本。

    参数:
      source_dna: 源物种编码 DNA 序列(CDS, 3 的倍数, ≤2100 bp)。
      source_species / target_species: OMA 物种 ID(如 IDESA/BACSU/ECOLI)。
      model_path: 预训练权重路径(aligned_model.pt 推荐, 更稳定; 缺省自动定位)。
      device: auto(按 torch.cuda.is_available 选) / cpu / cuda。
      beam_size: 1=greedy; >1 beam search(高 GC 序列稳健, 需 GPU)。
      fixed_aa_len: >0 时**固定生成长度** = 该氨基酸数, 忽略 EOS 并自动补终止子
        (用于高 GC 源序列模型「不生成 EOS」的退化场景 —— 实测前 N 密码子翻译
        与源蛋白 100% 一致, 仅缺失终止子; 补 TAA 后即为完整可表达 CDS)。
    返回:
      {"predicted_dna": ..., "predicted_protein": ...,
       "source_protein": ..., "target_species": ..., "source_species": ...}
    """
    import torch
    import torch.nn.functional as F
    from Bio.Seq import Seq

    t2i, i2t = _load_vocab(vocab_path)
    for sp in (source_species, target_species):
        if sp not in t2i:
            raise ValueError(
                f"物种 '{sp}' 不在 OrthologTransformer 的 2138 物种列表; "
                f"可用物种见 ortholog_vocab.json(如 IDESA/BACSU/ECOLI/SACVD)。")

    model_file = _resolve_model(model_path)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
        if device == "auto" else torch.device(device)

    t_start = time.time()
    LOG.info("[ortholog] 开始 %s→%s: 源 CDS %d bp, device=%s, beam_size=%d, "
             "fixed_aa_len=%d", source_species, target_species, len(source_dna),
             dev, beam_size, fixed_aa_len)

    vocab_size_target = 3 + 64 + 1  # <pad>/<bos>/<eos> + 64 密码子 + gap '---'
    vocab_size_input = len(t2i)
    CodonTransformer = _codon_transformer_cls()
    model = CodonTransformer(
        vocab_size_target, vocab_size_input,
        d_model=_D_MODEL, nhead=_NHEAD,
        num_encoder_layers=_N_ENC, num_decoder_layers=_N_DEC,
        dim_feedforward=_DIM_FF, dropout=_DROPOUT).to(dev)
    try:
        t_load = time.time()
        model.load_state_dict(torch.load(model_file, map_location=dev), strict=True)
        LOG.info("[ortholog] 权重加载完成 (%.1fs)", time.time() - t_load)
    except RuntimeError as e:
        raise RuntimeError(
            f"权重 {model_file} 与模型超参不符(期望 d_model={_D_MODEL} "
            f"nhead={_NHEAD} dim_feedforward={_DIM_FF} encoder/decoder="
            f"{_N_ENC}/{_N_DEC})。请用官方 aligned_model.pt / unaligned_model.pt。"
            f"\n原始错误: {e}") from e
    model.eval()

    src_idx = t2i[source_species]
    tgt_idx = t2i[target_species]
    bos, eos = t2i["<bos>"], t2i["<eos>"]

    src_tokens = _dna_to_tokens(source_dna, src_idx, t2i)
    src = torch.tensor([src_tokens], dtype=torch.long, device=dev)

    # 生成: fixed_aa_len>0 → 固定长度(退化场景); 否则 beam_size>1 beam / =1 greedy。
    dec_ipt = torch.tensor([[tgt_idx, bos]], dtype=torch.long, device=dev)
    t_gen = time.time()
    with torch.no_grad():
        if fixed_aa_len > 0:
            # 固定生成 fixed_aa_len 个密码子, 不依赖 EOS(高 GC 源序列常退化不产 EOS)
            for step in range(fixed_aa_len):
                logits = model(dec_ipt, src)
                probs = F.softmax(logits, dim=2)
                nxt = torch.argmax(probs, dim=2)[:, -1].unsqueeze(1)
                dec_ipt = torch.cat((dec_ipt, nxt), dim=1)
                if (step + 1) % 50 == 0:
                    LOG.info("[ortholog] 生成进度 %d/%d 密码子 (累计 %.1fs, 约 %.0f ms/步)",
                             step + 1, fixed_aa_len, time.time() - t_gen,
                             (time.time() - t_gen) / (step + 1) * 1000)
        elif beam_size > 1:
            LOG.info("[ortholog] beam search beam_size=%d (逐 batch forward)", beam_size)
            dec_ipt = _beam_search(model, src, dec_ipt, eos, beam_size, max_len)
        else:
            for step in range(max_len):
                logits = model(dec_ipt, src)
                probs = F.softmax(logits, dim=2)
                nxt = torch.argmax(probs, dim=2)[:, -1].unsqueeze(1)
                dec_ipt = torch.cat((dec_ipt, nxt), dim=1)
                if (dec_ipt == eos).any(dim=1).all():
                    LOG.info("[ortholog] 生成 %d 步后命中 EOS (%.1fs)",
                             step + 1, time.time() - t_gen)
                    break
                if (step + 1) % 50 == 0:
                    LOG.info("[ortholog] 生成进度 %d/%d 步 (累计 %.1fs, 约 %.0f ms/步)",
                             step + 1, max_len, time.time() - t_gen,
                             (time.time() - t_gen) / (step + 1) * 1000)
    LOG.info("[ortholog] 生成完成 (%.1fs)", time.time() - t_gen)

    pred_tokens = [int(t) for t in dec_ipt[0].tolist()]
    pred_dna = _tokens_to_dna(pred_tokens, i2t)
    if fixed_aa_len > 0:
        pred_dna = pred_dna[:fixed_aa_len * 3] + "TAA"  # 截断 + 补 E.coli 终止子
    src_dna = _tokens_to_dna(src_tokens, i2t)

    pred_protein = str(Seq(pred_dna).translate(to_stop=True))
    src_protein = str(Seq(src_dna).translate(to_stop=True))

    out = {
        "predicted_dna": pred_dna,
        "predicted_protein": pred_protein,
        "source_protein": src_protein,
        "target_species": target_species,
        "source_species": source_species,
        "n_predicted_bp": len(pred_dna),
        "n_source_bp": len(src_dna),
        "beam_size": beam_size,
        "fixed_aa_len": fixed_aa_len,
    }
    LOG.info("OrthologTransformer 重设计: %s→%s (%d bp → %d bp, 总耗时 %.1fs)",
             source_species, target_species, len(src_dna), len(pred_dna),
             time.time() - t_start)
    return out


__all__ = ["redesign", "_load_vocab"]
