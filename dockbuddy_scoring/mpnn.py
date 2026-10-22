# -*- coding: utf-8 -*-
"""
scoring.mpnn —— ProteinMPNN / LigandMPNN 反向折叠重设计适配器(2026-10-11 新增)
===================================================================================
定位: 候选突变生成器之一(C 类「序列重设计」) —— 用 ProteinMPNN(Dauparas et al.
2022, Science)在给定骨架上做反向折叠, 重设计指定区域(如底物沟)的残基, 固定催化
三联体等关键位点, 产出的新序列 diff WT 得突变清单。变体 LigandMPNN 显式建模配体
原子, 对「扩口袋/提升底物结合」更对口。

接入方式(独立 repo 薄适配器, 与 geostab/evmutation 同款):
  - 仓库 github.com/dauparas/ProteinMPNN(含 model_weights/, ~几 MB)。
  - 依赖: torch; 真跑需 GPU 实例 + repo + 权重。本机未部署。

诚实边界:
  - 本适配器是官方 repo 的调用包装, 不重实现模型; 缺 repo 时抛 RuntimeError。
  - ProteinMPNN 输出「重设计后的完整序列」(固定位点不变), 由下游 diff 回 WT 得
    突变清单; 采样温度/次数影响多样性, 需多次采样取稳定重复的替换。
  - 反向折叠只保证「序列能折叠回该骨架」, 不保证活性/稳定性提升, 是候选来源而非结论。
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Sequence, Tuple

LOG = logging.getLogger("dockpipeline")

_MPNN_HINT = (
    "ProteinMPNN 重设计需要其官方仓库(github.com/dauparas/ProteinMPNN, 克隆后 "
    "--repo-dir 指向它) + torch + model_weights/(随仓)。本机未部署。请在 GPU 实例"
    "按官方 README 部署后运行。"
)

_AA_SET = set("ACDEFGHIKLMNPQRSTVWY")


def diff_sequences(wt_seq: str, new_seq: str,
                   redesign_sites: Sequence[int]) -> List[Tuple[int, str, str]]:
    """diff 重设计后的序列, 返回 [(site 1-based, wt_aa, new_aa), ...]。

    只对比 redesign_sites 内的位点(固定位点不在 diff 范围); 越界/非标准残基跳过。
    """
    out: List[Tuple[int, str, str]] = []
    for site in redesign_sites:
        i = site - 1
        if i >= len(new_seq) or i >= len(wt_seq):
            continue
        wt = wt_seq[i].upper()
        new = new_seq[i].upper()
        if wt in _AA_SET and new in _AA_SET and wt != new:
            out.append((site, wt, new))
    return out


def redesign_sequences(wt_seq: str, structure_pdb: str,
                       fixed_sites: Sequence[int], redesign_sites: Sequence[int], *,
                       repo_dir: str = "", num_seqs: int = 1,
                       temperature: float = 0.1, seed: int = 0,
                       checkpoint: str = "") -> Dict[str, Any]:
    """用 ProteinMPNN 重设计指定区域, 返回 {sequences: [new_seq, ...], wt_seq}。

    参数:
      wt_seq: WT 序列(1-letter)。
      structure_pdb: WT 结构 PDB(绝对路径)。
      fixed_sites: 固定残基位点(1-based, 如催化三联体 126/174/204, 保持 WT)。
      redesign_sites: 重设计位点(1-based, 如底物沟残基)。实现上把「非 redesign 的
        位点」全部固定, 故只有 redesign_sites 会被反向折叠重设计。
      repo_dir: ProteinMPNN 仓库路径(必填)。
      num_seqs / temperature / seed: 采样数 / 采样温度 / 随机种子。
      checkpoint: 权重路径(缺省 repo_dir/vanilla_model_weights/v_48_020.pt;
        可指定 ThermoMPNN/Soluble 等变体权重)。
    """
    import copy
    import os
    import sys

    if repo_dir and repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)
    try:
        import numpy as np
        import torch
        from protein_mpnn_utils import (ProteinMPNN, StructureDatasetPDB,
                                        _S_to_seq, parse_PDB, tied_featurize)
    except ImportError as e:
        raise RuntimeError(_MPNN_HINT) from e

    torch.manual_seed(seed)
    device = "cpu"

    # 1. 解析 PDB
    pdb_dict_list = parse_PDB(structure_pdb, ca_only=False)
    dataset = StructureDatasetPDB(pdb_dict_list, truncate=None, max_length=100000)
    all_chains = [k[-1:] for k in list(pdb_dict_list[0]) if k[:9] == "seq_chain"]
    pdb_name = pdb_dict_list[0]["name"]

    # 2. 加载模型(默认 vanilla v_48_020; 可指定 checkpoint)
    checkpoint_path = checkpoint or os.path.join(repo_dir, "vanilla_model_weights", "v_48_020.pt")
    if not os.path.isfile(checkpoint_path):
        raise RuntimeError(f"ProteinMPNN 权重缺失: {checkpoint_path}")
    # torch>=2.6 默认 weights_only=True 拦截 ProteinMPNN 官方 checkpoint(含
    # argparse.Namespace); 权重为官方发布(v_48_020.pt), 属可信来源。
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:  # torch<1.13 无此参数
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    # 兼容两种格式: vanilla 用 model_state_dict, Lightning(如 ThermoMPNN)用 state_dict
    state_dict = checkpoint.get("model_state_dict") or checkpoint.get("state_dict")
    if state_dict is None:
        raise RuntimeError(f"ProteinMPNN 权重格式未知(无 model_state_dict/state_dict): {checkpoint_path}")
    # 剥离 Lightning 包装前缀(如 ThermoMPNN 的 "model.prot_mpnn.")。必须**条件式**剥离
    # (只剥匹配前缀的 key): ThermoMPNN 的 state_dict 混有 prot_mpnn.*(backbone) 和
    # light_attention.*(温度头) 两类 key, 若对所有 key 无脑 k[len("prot_mpnn."):] 会把
    # light_attention.* 截成 "ntion.*" 等脏 key → 既漏过滤又报 Unexpected key。
    def _strip_prefix(k):
        if k.startswith("model."):
            k = k[len("model."):]
        if k.startswith("prot_mpnn."):
            k = k[len("prot_mpnn."):]
        return k
    state_dict = {_strip_prefix(k): v for k, v in state_dict.items()}
    # 坑: ThermoMPNN 权重在 ProteinMPNN backbone 之外还挂了 3 个预测头 ——
    #   light_attention.*(温度预测)、both_out.*(ddG+Tm 双输出)、ddg_out.*(ΔΔG 头)。
    # 序列重设计只需 ProteinMPNN backbone, 故过滤掉这 3 个头, 否则 load_state_dict
    # 报 "Unexpected key(s)"。注意 W_out 是标准 ProteinMPNN 输出层, 不能过滤。
    _PRED_HEADS = ("light_attention.", "both_out.", "ddg_out.")
    state_dict = {k: v for k, v in state_dict.items()
                  if not k.startswith(_PRED_HEADS)}
    hidden_dim, num_layers = 128, 3
    model = ProteinMPNN(num_letters=21, node_features=hidden_dim,
                        edge_features=hidden_dim, hidden_dim=hidden_dim,
                        num_encoder_layers=num_layers, num_decoder_layers=num_layers,
                        augment_eps=0.05, k_neighbors=checkpoint.get("num_edges", 48))
    model.load_state_dict(state_dict)
    model.eval()

    # 3. 固定掩码: 只有 redesign_sites 可被重设计(其余全固定)
    redesign_set = set(int(s) for s in redesign_sites)
    fixed_all = [i for i in range(1, len(wt_seq) + 1) if i not in redesign_set]
    chain_id_dict = {pdb_name: (all_chains, [])}
    fixed_positions_dict = {pdb_name: {letter: fixed_all for letter in all_chains}}

    alphabet = "ACDEFGHIKLMNPQRSTVWYX"
    omit_AAs_np = np.zeros(21, dtype=np.float32)
    bias_AAs_np = np.zeros(21, dtype=np.float32)

    sequences: List[str] = []
    with torch.no_grad():
        for protein in dataset:
            batch_clones = [copy.deepcopy(protein) for _ in range(num_seqs)]
            (X, S, mask, lengths, chain_M, chain_encoding_all, chain_list_list,
             visible_list_list, masked_list_list, masked_chain_length_list_list,
             chain_M_pos, omit_AA_mask, residue_idx, dihedral_mask,
             tied_pos_list, pssm_coef, pssm_bias, pssm_log_odds_all,
             bias_by_res_all, tied_beta) = tied_featurize(
                batch_clones, device, chain_id_dict, fixed_positions_dict,
                None, None, None, None, ca_only=False)
            randn = torch.randn(chain_M.shape, device=device)
            for i in range(num_seqs):
                sample = model.sample(
                    X, randn, S, chain_M, chain_encoding_all, residue_idx,
                    mask=mask, temperature=temperature, omit_AAs_np=omit_AAs_np,
                    bias_AAs_np=bias_AAs_np, chain_M_pos=chain_M_pos,
                    omit_AA_mask=omit_AA_mask, bias_by_res=bias_by_res_all)
                sequences.append(_S_to_seq(sample["S"][i], mask[i]))

    return {"sequences": sequences, "wt_seq": wt_seq,
            "fixed_sites": list(fixed_sites), "redesign_sites": list(redesign_sites)}
