# -*- coding: utf-8 -*-
"""FoldX 5.1 集成 —— 真 ΔΔG 稳定性预测 (替代 ddg.py 的禁用 proxy)
=====================================================================

调用链 (FoldX 5.1 实测验证, 见 data/reports/rational_design_roadmap.md)
----------------------------------------------------------------
1. RepairPDB  : 补全侧链 / 修复结构 -> <stem>_Repair.pdb + <stem>_Repair.fxout
2. BuildModel : 对修复后结构做突变建模 -> 产出突变体 PDB 与能量文件
   - 突变文件**必须命名为 individual_list.txt**, 且每行突变以分号 ';' 结尾,
     例如:  `FA205G;`  (否则 BuildModel 报 "I did not find the wild-type
     DNA or PROTEIN sequence in the X-tal structure" 或静默不建模)。
     注: 文件名 mutant_file*.txt 虽满足 --help 的前缀检查, 但 BuildModel 实际
     只正确解析 individual_list.txt, 已实测验证。
   - 突变格式: <WT残基1字母><chain><pos><MUT残基1字母>, 例如 FA205G
   - --pdb 是相对文件名, 须配合 --pdb-dir; rotabase.txt 须与二进制同目录(cwd)
   - BuildModel 会**内部**为每个突变构建对应的 WT 参照并直接给出 ΔΔG,
     无需外部再跑一次 no-op 参照。

能源解析 (FoldX 5.1 实测产物命名)
--------------------------------
BuildModel 在输出目录生成四个 fxout:
  Dif_<stem>.fxout      突变体相对其 WT 参照的能量差 = ΔΔG (关键)
  Average_<stem>.fxout  各 run 平均能量
  Raw_<stem>.fxout      突变体 + WT 参照的完整能量分解 (含各自总能量)
  PdbList_<stem>.fxout  突变体 PDB 与其 WT 参照 PDB 的对应关系
本模块从 Dif_<stem>.fxout 取 "total energy" 列作为 ΔΔG;
可选地再从 Raw_<stem>.fxout 取突变体 / WT 绝对能量回填 wt_energy / mut_energy。
单结构 ΔΔG 非自由能均值; 多突变取 mean±std 才报。

已知坑 (已实测固化, 见下方函数注释)
----------------------------------
- 突变文件名必须为 individual_list.txt, 且每行以 ';' 结尾 (最关键的坑)。
- 先 RepairPDB 再 BuildModel; BuildModel 内部会再修一次, 这是正常的。
- AF 预测模型 (Open Babel 转出的无氢 PDB) 在正确格式下可正常跑通,
  无需额外补氢 / SEQRES / TER (已用 1CRN 晶体正对照 + PBATase m4 AF 模型验证)。
- 错误签名: "Specified residue not found" 表示突变位点残基与结构不符
  (检查 WT 残基 / 链 / 编号); "did not find the wild-type sequence" 表示
  突变文件格式错误(未用 individual_list.txt 或缺少 ';')。

诚实口径 (沿用项目纪律)
----------------------
- FoldX ΔΔG 是真实热力学代理(含 unfolded 态参考 + 系综思想), 远优于 ddg.py 的
  噪声地板 proxy; 但仍须**先做 1~2 个有实验 ΔΔG 的突变体做方法学标定**再写入论文。
- 禁止把单点 ΔΔG 直接推 "活性升降"; 机制结论仍须湿实验(kcat/KM, 酰基酶质谱)收口。
- 单结构 ΔΔG 非自由能均值; 多 run / 多突变取 mean±std 才报。
"""
from __future__ import annotations

import glob
import os
import re
import shutil
import subprocess
import tempfile
from typing import Dict, List, Optional, Tuple

__all__ = [
    "FOLDX_BIN_ENV",
    "find_foldx_bin",
    "foldx_ddg",
    "foldx_build_model",
    "FoldXError",
]

FOLDX_BIN_ENV = "FOLDX_BIN"
# 不在代码里硬编码个人桌面路径(保持可移植); 二进制位置由 FOLDX_BIN 环境变量或
# CLI --bin 指定。常见候选仅作回退探测(均不指向具体用户目录)。
_FOLDX_BIN_CANDIDATES: Tuple[str, ...] = (
    "foldx",        # Linux 软链(/usr/local/bin/foldx, 部署侧 tool_install 落位)
    "foldx_1.exe",
    "foldx.exe",
    "foldx5.exe",
)

# BuildModel 突变文件: 必须此名(实测, mutant_file* 不被正确解析)。
MUTANT_LIST_NAME = "individual_list.txt"


class FoldXError(RuntimeError):
    """FoldX 调用失败(二进制缺失 / 运行报错 / 能量解析失败)。"""


def find_foldx_bin(explicit: Optional[str] = None) -> Optional[str]:
    """定位 FoldX 5.1 二进制。

    优先级: 显式参数 > FOLDX_BIN 环境变量 > PATH/常见名探测。
    找不到返回 None(由调用方决定短路还是报错)。
    """
    if explicit:
        if not os.path.isfile(explicit):
            raise FoldXError(f"指定的 FoldX 二进制不存在: {explicit}")
        return explicit
    env = os.environ.get(FOLDX_BIN_ENV, "").strip()
    if env:
        if not os.path.isfile(env):
            raise FoldXError(f"{FOLDX_BIN_ENV} 指向的二进制不存在: {env}")
        return env
    for name in _FOLDX_BIN_CANDIDATES:
        found = shutil.which(name)
        if found:
            return found
    return None


_TOTAL_RE = re.compile(r"Total\s*=\s*(-?\d+(?:\.\d+)?)")


def _parse_total(text: str) -> Optional[float]:
    """从文本提取最后一个 'Total = <val>'(RepairPDB fxout 用)。"""
    vals = _TOTAL_RE.findall(text)
    if not vals:
        return None
    return float(vals[-1])


_MUT_RE = re.compile(r"^([A-Za-z])([A-Za-z]?)(\d+)([A-Za-z])$")


def _normalize_mutation(mut: str, chain: str) -> str:
    """把 'F205G' 或 'FA205G' 归一为 FoldX 格式 '<WT><chain><pos><MUT>'。

    - 无 chain(如 F205G, 5 字符): 补 chain -> 'FA205G'
    - 已有 chain(如 FA205G, 6 字符): 校验 chain 一致性
    """
    mut = mut.strip()
    m = _MUT_RE.match(mut)
    if not m:
        raise FoldXError(
            f"无法解析突变 '{mut}'; 期望格式 '<WT残基><pos><MUT残基>'(如 F205G) "
            f"或 '<WT残基><chain><pos><MUT残基>'(如 FA205G)"
        )
    wt, ch, pos, mt = m.group(1), m.group(2), m.group(3), m.group(4)
    if ch:  # 已带链
        if ch != chain:
            raise FoldXError(
                f"突变 '{mut}' 的链 '{ch}' 与 --chain '{chain}' 不一致"
            )
        return mut
    return f"{wt}{chain}{pos}{mt}"


def _run(bin_path: str, args: List[str], cwd: str, timeout: int) -> Tuple[int, str]:
    """运行 FoldX, 返回 (returncode, 合并输出)。cwd 须含 rotabase.txt。"""
    try:
        proc = subprocess.run(
            [bin_path, *args],
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:  # pragma: no cover - 超时
        raise FoldXError(f"FoldX 超时(>{timeout}s): {' '.join(args)}") from exc
    out = proc.stdout or ""
    return proc.returncode, out


# BuildModel 已知错误签名(命中即判定失败)。
_ERROR_SIGS = (
    "Foldx will end",
    "No pdbs for the run",
    "did not find the wild-type",
    "sorry your filenames should start",
    "unrecognised option",
    "Specified residue not found",
)


def _parse_foldx_table(path: str) -> Optional[List[Tuple[str, float]]]:
    """解析 FoldX fxout 表格(Repair/Dif/Raw 同构): 返回 [(pdb名, total energy), ...]。

    fxout 头部为 FoldX banner, 随后 'Pdb\\ttotal energy\\t...' 表头行,
    其后是数据行。取 'Pdb' 列与 'total energy' 列。
    Dif 的 total energy = ΔΔG; Raw 的 total energy = 绝对能量。
    """
    try:
        with open(path, encoding="utf-8", errors="ignore") as fh:
            lines = fh.read().splitlines()
    except OSError:
        return None
    hdr_idx = None
    for i, line in enumerate(lines):
        if line.startswith("Pdb"):
            hdr_idx = i
            break
    if hdr_idx is None:
        return None
    hdr = lines[hdr_idx].split("\t")
    try:
        ni = hdr.index("Pdb")
        ti = hdr.index("total energy")
    except ValueError:
        return None
    rows: List[Tuple[str, float]] = []
    for line in lines[hdr_idx + 1:]:
        if not line.strip():
            continue
        cols = line.split("\t")
        if len(cols) <= max(ni, ti):
            continue
        name = cols[ni].strip()
        try:
            val = float(cols[ti])
        except ValueError:
            continue
        rows.append((name, val))
    return rows


def _parse_dif_ddg(path: str) -> Optional[float]:
    """从 Dif_<stem>.fxout 取 ΔΔG (total energy 列, 取首个突变行)。"""
    rows = _parse_foldx_table(path)
    if not rows:
        return None
    return rows[0][1]


def _parse_raw_wt_mut(path: str) -> Tuple[Optional[float], Optional[float]]:
    """从 Raw_<stem>.fxout 取 (wt_energy, mut_energy)。

    Raw 含两行: 突变体(名不含 'WT_' 前缀)与 WT 参照(名以 'WT_' 开头)。
    """
    rows = _parse_foldx_table(path)
    if not rows:
        return None, None
    wt = None
    mut = None
    for name, val in rows:
        if name.startswith("WT_"):
            wt = val
        else:
            mut = val
    return wt, mut


def _repair(
    bin_path: str,
    cwd: str,
    pdb_dir: str,
    pdb_name: str,
    workdir: str,
    timeout: int,
) -> Tuple[str, Optional[float]]:
    """RepairPDB: 返回 (修复后 pdb 绝对路径, 修复总能量)。"""
    args = [
        "--command=RepairPDB",
        f"--pdb-dir={pdb_dir}",
        f"--pdb={pdb_name}",
        f"--output-dir={workdir}",
    ]
    rc, out = _run(bin_path, args, cwd, timeout)
    if rc != 0:
        raise FoldXError(f"RepairPDB 失败 (rc={rc}): {out[-2000:]}")
    repaired = os.path.join(workdir, f"{os.path.splitext(pdb_name)[0]}_Repair.pdb")
    if not os.path.isfile(repaired):
        raise FoldXError(f"RepairPDB 未产出修复 PDB: {repaired}\n{out[-2000:]}")
    fxout = os.path.join(workdir, f"{os.path.splitext(pdb_name)[0]}_Repair.fxout")
    total = _parse_total(out) if os.path.isfile(fxout) else None
    return repaired, total


def _buildmodel(
    bin_path: str,
    cwd: str,
    pdb_dir: str,
    workdir: str,
    repaired_name: str,
    mutation: str,
    timeout: int,
) -> Tuple[Optional[float], Optional[str], Dict[str, Optional[float]]]:
    """对单个已归一突变跑 BuildModel, 返回 (ddg, 错误提示, {wt,mut} 能量)。

    关键修正 (vs 早期实现):
    - 突变文件必须命名为 individual_list.txt, 且内容以 ';' 结尾 (如 'FA205G;')。
      早期 'mutant_file.txt' 或缺少 ';' 会触发 BuildModel 报
      "I did not find the wild-type DNA or PROTEIN sequence ..."。
    - 能量从 Dif_<stem>.fxout 的 'total energy' 列解析 (= ΔΔG),
      不再从 stdout 的 'Total ='(那是内部 Repair 预步骤能量, 失败也会打印,
      曾导致 wt=mut 的假阳性 ΔΔG=0)。
    - BuildModel 内部为每个突变构建 WT 参照, 故单次运行即得 ΔΔG,
      无需外部再跑 no-op 参照。
    """
    list_path = os.path.join(workdir, MUTANT_LIST_NAME)
    with open(list_path, "w", encoding="utf-8") as fh:
        fh.write(f"{mutation};\n")
    args = [
        "--command=BuildModel",
        f"--pdb-dir={pdb_dir}",
        f"--pdb={repaired_name}",
        f"--mutant-file={list_path}",
        f"--output-dir={workdir}",
    ]
    rc, out = _run(bin_path, args, cwd, timeout)
    # 错误签名检测(即便 rc 被包装成 127 也要透传真实原因)
    for sig in _ERROR_SIGS:
        if sig in out:
            return None, _classify_error(sig, out), {}
    # 解析 Dif_*.fxout (取本次新生成的; 单突变应只有一个)
    dif_files = sorted(glob.glob(os.path.join(workdir, "Dif_*.fxout")))
    ddg = None
    if dif_files:
        ddg = _parse_dif_ddg(dif_files[-1])
    if ddg is None:
        return None, "BuildModel 未产出可解析的 Dif fxout(ΔΔG 缺失)", {}
    # 可选: Raw 取 WT / 突变绝对能量
    raw_files = sorted(glob.glob(os.path.join(workdir, "Raw_*.fxout")))
    wt_e = mut_e = None
    if raw_files:
        wt_e, mut_e = _parse_raw_wt_mut(raw_files[-1])
    return ddg, None, {"wt_energy": wt_e, "mut_energy": mut_e}


def _classify_error(sig: str, out: str) -> str:
    """把 FoldX 错误签名翻译成可操作的提示(透传, 不静默)。"""
    if "Specified residue not found" in sig:
        return ("突变位点残基与结构不符: 检查 WT 残基 / 链 / 编号是否与 PDB 一致。"
                f" 原始片段: {out[-300:].strip()}")
    if "did not find the wild-type" in sig:
        return ("FoldX 无法识别野生型序列: 突变文件须命名为 individual_list.txt "
                "且每行突变以 ';' 结尾(如 'FA205G;')。原始片段: "
                f"{out[-300:].strip()}")
    if "sorry your filenames should start" in sig:
        return f"突变文件名格式错误(须 individual_list.txt)。片段: {out[-300:].strip()}"
    if "unrecognised option" in sig:
        return f"FoldX 参数不被识别。片段: {out[-300:].strip()}"
    if "No pdbs for the run" in sig:
        return f"FoldX 未找到输入 PDB(pdb-dir 或 --pdb 有误)。片段: {out[-300:].strip()}"
    return f"FoldX 运行失败({sig})。片段: {out[-300:].strip()}"


def _read(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            return fh.read()
    except OSError:
        return ""


def foldx_ddg(
    receptor_pdb: str,
    mutation: str,
    *,
    bin_path: Optional[str] = None,
    chain: str = "A",
    workdir: str = "",
    pH: float = 7.0,
    temperature: float = 298.0,
    timeout: int = 600,
) -> Dict:
    """计算单点(或逗号分隔多点)突变的 ΔΔG (kcal/mol)。

    参数
    ----
    receptor_pdb : WT 结构 PDB(绝对路径)
    mutation     : 如 'F205G' 或 'FA205G'(可逗号分隔多突变)
    bin_path     : FoldX 二进制; None 则走 find_foldx_bin()
    chain        : 突变所在链(默认 A)
    workdir      : 工作目录; 空则建临时目录

    Returns
    -------
    dict: {ok, available, ddG_kcal, ddG_std, method, wt_energy, mut_energy,
           repair_energy, runs, note, verdict{ok,narrative,banned_phrases,
           allowed_phrases}}
    """
    # ---- 前置: 二进制 + 输入 ----
    bin_path = find_foldx_bin(bin_path)
    if bin_path is None:
        return {
            "ok": False,
            "available": False,
            "ddG_kcal": None,
            "ddG_std": None,
            "method": "FoldX5.1 BuildModel",
            "wt_energy": None,
            "mut_energy": None,
            "repair_energy": None,
            "runs": 0,
            "note": "未找到 FoldX 二进制; 设置环境变量 FOLDX_BIN 或传 --bin",
            "verdict": {
                "ok": False,
                "narrative": "FoldX 不可用: 未配置二进制 (FOLDX_BIN)。",
                "banned_phrases": [],
                "allowed_phrases": [],
            },
        }
    if not os.path.isfile(receptor_pdb):
        raise FoldXError(f"受体 PDB 不存在: {receptor_pdb}")

    # cwd = 二进制所在目录(保证 rotabase.txt 可见)
    cwd = os.path.dirname(os.path.abspath(bin_path))
    cleanup = False
    if not workdir:
        workdir = tempfile.mkdtemp(prefix="foldx_")
        cleanup = True
    else:
        # 调用方显式传 workdir 时须确保目录存在(此前缺失 → shutil.copy 报
        # FileNotFoundError; 2026-10-10 云实例批量打分暴露并修复)
        os.makedirs(workdir, exist_ok=True)
    try:
        # 复制输入 PDB 到工作目录(便于 --pdb-dir 引用)
        pdb_name = os.path.basename(receptor_pdb)
        shutil.copy(receptor_pdb, os.path.join(workdir, pdb_name))

        # 归一化每个突变
        muts = [m.strip() for m in mutation.split(",") if m.strip()]
        if not muts:
            raise FoldXError(f"突变列表为空: '{mutation}'")
        norms = [_normalize_mutation(m, chain) for m in muts]

        # ---- 1. RepairPDB (一次, 后续 BuildModel 内部会复用/再修) ----
        try:
            repaired, repair_energy = _repair(
                bin_path, cwd, workdir, pdb_name, workdir, timeout
            )
        except FoldXError as exc:
            return _fail(muts, None, None, repair_energy=None,
                         note=f"RepairPDB 失败: {exc}")
        repaired_name = os.path.basename(repaired)

        # ---- 2. BuildModel (每个突变独立运行, 单次即得 ΔΔG) ----
        ddgs: List[float] = []
        wt_energies: List[float] = []
        mut_energies: List[float] = []
        err_notes: List[str] = []
        for norm in norms:
            # 每次用独立子目录, 避免 Dif/Raw fxout 互相覆盖
            sub = os.path.join(workdir, f"bm_{norm}")
            os.makedirs(sub, exist_ok=True)
            # BuildModel 需要 repaired PDB 在 pdb-dir 中
            shutil.copy(repaired, os.path.join(sub, repaired_name))
            ddg, err, info = _buildmodel(
                bin_path, cwd, sub, sub, repaired_name, norm, timeout
            )
            if err is not None or ddg is None:
                err_notes.append(f"{norm}: {err or 'ΔΔG 解析失败'}")
                ddgs.append(None)
            else:
                ddgs.append(ddg)
                if info.get("wt_energy") is not None:
                    wt_energies.append(info["wt_energy"])
                if info.get("mut_energy") is not None:
                    mut_energies.append(info["mut_energy"])

        if any(d is None for d in ddgs):
            return _fail(
                muts, None, None, repair_energy=repair_energy,
                note="; ".join(err_notes) or "部分突变未能计算 ΔΔG",
            )

        valid = [d for d in ddgs if d is not None]
        mean_ddg = sum(valid) / len(valid)
        if len(valid) > 1:
            var = sum((d - mean_ddg) ** 2 for d in valid) / (len(valid) - 1)
            std_ddg = var ** 0.5
        else:
            std_ddg = 0.0
        mean_wt = _mean(wt_energies)
        mean_mut = _mean(mut_energies)

        return {
            "ok": True,
            "available": True,
            "ddG_kcal": round(mean_ddg, 4),
            "ddG_std": round(std_ddg, 4),
            "method": "FoldX5.1 BuildModel",
            "wt_energy": (round(mean_wt, 4) if mean_wt is not None else None),
            "mut_energy": (round(mean_mut, 4) if mean_mut is not None else None),
            "repair_energy": (round(repair_energy, 4)
                              if repair_energy is not None else None),
            "runs": len(valid),
            "note": (
                "单 run ΔΔG(均值); 多突变取 mean±std。写入论文前须用 1~2 个有实验 "
                "ΔΔG 的突变体标定。单点 ΔΔG 不得直接推断活性升降。"
            ),
            "verdict": {
                "ok": True,
                "narrative": (
                    f"FoldX 5.1 BuildModel 给出 {', '.join(norms)} 的 ΔΔG ≈ "
                    f"{mean_ddg:+.3f} kcal/mol"
                    + (f" (±{std_ddg:.3f}, n={len(valid)})" if len(valid) > 1 else "")
                    + f" (WT {mean_wt:.2f}, MUT {mean_mut:.2f}); "
                    f"此为稳定性代理, 须经实验 ΔΔG 标定且不得直接推活性。"
                ),
                "banned_phrases": [
                    "活性提升", "活性增强", "活性增高", "活性上升",
                    "ΔΔG = 精确", "自由能严格",
                ],
                "allowed_phrases": [
                    "稳定性代理", "ΔΔG 近似", "需实验标定",
                ],
            },
        }
    finally:
        if cleanup:
            shutil.rmtree(workdir, ignore_errors=True)


def _mean(vals: List[float]) -> Optional[float]:
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def foldx_build_model(
    receptor_pdb: str,
    mutation: str,
    *,
    bin_path: Optional[str] = None,
    chain: str = "A",
    out_pdb: str = "",
    timeout: int = 600,
) -> Dict:
    """RepairPDB + BuildModel 建模突变体, 返回突变体 PDB 路径 + ΔΔG(环节 7 建模)。

    与 foldx_ddg 同流程, 但额外暴露 BuildModel 生成的突变体 PDB(复制到 out_pdb,
    缺省 <receptor>_<mutation>.pdb)。
    """
    import glob as _glob

    bin_path = find_foldx_bin(bin_path)
    if bin_path is None:
        raise FoldXError("未找到 FoldX 二进制; 设置 FOLDX_BIN 或传 bin_path")
    if not os.path.isfile(receptor_pdb):
        raise FoldXError(f"受体 PDB 不存在: {receptor_pdb}")
    cwd = os.path.dirname(os.path.abspath(bin_path))
    workdir = tempfile.mkdtemp(prefix="foldx_bm_")
    try:
        pdb_name = os.path.basename(receptor_pdb)
        shutil.copy(receptor_pdb, os.path.join(workdir, pdb_name))
        norm = _normalize_mutation(mutation.strip(), chain)
        repaired, _repair_energy = _repair(bin_path, cwd, workdir, pdb_name,
                                           workdir, timeout)
        repaired_name = os.path.basename(repaired)
        sub = os.path.join(workdir, f"bm_{norm}")
        os.makedirs(sub, exist_ok=True)
        shutil.copy(repaired, os.path.join(sub, repaired_name))
        ddg, err, info = _buildmodel(bin_path, cwd, sub, sub, repaired_name,
                                     norm, timeout)
        if err is not None or ddg is None:
            raise FoldXError(f"BuildModel 失败: {err or 'ΔΔG 解析失败'}")

        # BuildModel 突变体 PDB(除 repaired 外的 *_1.pdb 等)
        cand = [p for p in _glob.glob(os.path.join(sub, "*.pdb"))
                if os.path.basename(p) != repaired_name]
        mutant_pdb = cand[0] if cand else ""
        out = out_pdb or receptor_pdb.rsplit(".", 1)[0] + f"_{mutation.strip()}.pdb"
        if mutant_pdb and os.path.isfile(mutant_pdb):
            if os.path.dirname(out):
                os.makedirs(os.path.dirname(out), exist_ok=True)
            shutil.copy(mutant_pdb, out)
        return {"ok": True, "mutation": norm, "ddG_kcal": round(ddg, 4),
                "mutant_pdb": out, "repaired_pdb": repaired,
                "wt_energy": info.get("wt_energy"),
                "mut_energy": info.get("mut_energy")}
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def _fail(mut, wt_energy, mut_energy, *, repair_energy=None, note: str) -> Dict:
    muts = ", ".join(mut) if isinstance(mut, (list, tuple)) else str(mut)
    return {
        "ok": False,
        "available": True,
        "ddG_kcal": None,
        "ddG_std": None,
        "method": "FoldX5.1 BuildModel",
        "wt_energy": (round(wt_energy, 4) if wt_energy is not None else None),
        "mut_energy": (round(mut_energy, 4) if mut_energy is not None else None),
        "repair_energy": (round(repair_energy, 4)
                          if repair_energy is not None else None),
        "runs": 0,
        "note": note,
        "verdict": {
            "ok": False,
            "narrative": f"FoldX ΔΔG 计算未成功 ({muts}): {note}",
            "banned_phrases": [],
            "allowed_phrases": [],
        },
    }


def _tail(text: str, n: int = 600) -> str:
    return text[-n:].strip()
