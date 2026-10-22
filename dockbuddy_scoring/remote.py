# -*- coding: utf-8 -*-
"""
libscore_remote.py —— 远程 GPU 打分通道(腾讯云 TAT, 2026-10-08)
===================================================================================
定位: 预筛管线「打分」步骤的**远程真实调用**通道 —— 本机无 GPU 时, 通过腾讯云 TAT
(自动化助手, 免 SSH 免密码)把 `run_batch.py` 打分命令发到 GPU 实例, 轮询拿回 scores,
实现「流水线调用 → 真实模型输入输出」端到端走通。

用法(由 orchestrator 的 scoring.remote 触发, 见 run_pipeline):
  from scoring.remote import score_variants
  scores = score_variants("esm1v", wt_seq, variants, remote_cfg)
  # -> {"F205G": 1.234, ...}   (值为模型打分, metric 名由 orchestrator 统一包装)

remote_cfg 字段:
  transport   "tat"(目前唯一)
  secret_id / secret_key   腾讯云密钥(最小权限: tat + cvm 只读即可)
  region      地域(如 ap-nanjing)
  instance_id 目标 GPU 实例
  work_dir    实例上 run_batch.py 所在目录(缺省 /work)
  models_dir / model_path / device   透传给 run_batch.py
  structure_path / chain   saprot 专用: WT 结构 PDB/CIF 与链(结构感知打分必需)
  timeout     轮询总超时秒(缺省 1800)

设计决策(诚实):
  - **零依赖 import**: tencentcloud SDK 延迟 import; 本机未装 SDK 时抛 RuntimeError。
  - **DNS 绕过**: 复用本机坏 DNS 的 workaround(8.8.8.8 手动解析 *.tencentcloudapi.com),
    只在 transport=tat 且系统 DNS 解析失败时启用; 对非腾讯云域名不干预。
  - **stdout 契约**: 只抓 run_batch.py 输出的**最后一行 JSON**; 其余日志走 stderr。
  - 打分只是代理指标, 不预测 kcat/Km。
"""
from __future__ import annotations

import base64
import json
import logging
import time
from typing import Any, Dict, Iterable, Optional

LOG = logging.getLogger("dockpipeline")

_TAT_HINT = (
    "远程 GPU 打分需要 tencentcloud-sdk-python(tat 服务) + 可用的腾讯云 GPU 实例; "
    "本机未装 SDK 或未配置 remote 参数。安装: pip install tencentcloud-sdk-python; "
    "或改用本地 scoring.adapter=scores_file/foldx。"
)


# ---------------------------------------------------------------------------
# DNS 绕过(本机系统 DNS 坏时, 对 *.tencentcloudapi.com 用 8.8.8.8 手动解析)
# —— 单一事实源在 _net.py, 此处保留 _patch_dns 别名以兼容既有调用。
# ---------------------------------------------------------------------------
from _net import patch_dns as _patch_dns  # noqa: E402,F401


# ---------------------------------------------------------------------------
# TAT client(延迟 import)
# ---------------------------------------------------------------------------
def _tat_client(cfg: Dict[str, Any]):
    import tencentcloud  # noqa: F401  (触发 SDK 存在性检查)
    from tencentcloud.common import credential
    from tencentcloud.common.profile.client_profile import ClientProfile
    from tencentcloud.common.profile.http_profile import HttpProfile
    from tencentcloud.tat.v20201028 import tat_client, models as _m

    cred = credential.Credential(
        str(cfg.get("secret_id", "")), str(cfg.get("secret_key", "")))
    hp = HttpProfile()
    hp.endpoint = "tat.tencentcloudapi.com"
    cp = ClientProfile()
    cp.httpProfile = hp
    client = tat_client.TatClient(cred, str(cfg.get("region", "ap-nanjing")), cp)
    return client, _m


def _run_command(client, models, cfg: Dict[str, Any], cmd: str, timeout: int) -> str:
    req = models.RunCommandRequest()
    req.Content = base64.b64encode(cmd.encode("utf-8")).decode("ascii")
    req.InstanceIds = [str(cfg["instance_id"])]
    req.CommandType = "SHELL"
    req.Timeout = timeout
    resp = client.RunCommand(req)
    return resp.InvocationId


def _wait_result(client, models, inv_id: str, timeout: int) -> Optional[str]:
    """轮询到 SUCCESS/FAILED/TIMEOUT, 返回解码后的 stdout(或 None)。"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        time.sleep(6)
        r = models.DescribeInvocationsRequest()
        r.InvocationIds = [inv_id]
        invs = client.DescribeInvocations(r).InvocationSet or []
        if not invs:
            continue
        st = invs[0].InvocationStatus
        if st not in ("SUCCESS", "FAILED", "TIMEOUT"):
            continue
        task_ids = [t.InvocationTaskId for t in (invs[0].InvocationTaskBasicInfoSet or [])]
        if not task_ids:
            return None
        r2 = models.DescribeInvocationTasksRequest()
        r2.InvocationTaskIds = task_ids
        resp2 = client.DescribeInvocationTasks(r2)
        out = ""
        for t in (resp2.InvocationTaskSet or []):
            tr = t.TaskResult
            if tr and tr.Output:
                try:
                    out += base64.b64decode(tr.Output).decode("utf-8", errors="replace")
                except Exception:  # noqa: BLE001
                    out += str(tr.Output)
            if t.TaskStatus == "FAILED":
                LOG.warning("TAT 任务失败: %s (ExitCode=%s)", t.TaskStatus,
                            (tr.ExitCode if tr else None))
        return out or None
    LOG.warning("远程打分轮询超时(>%ss)", timeout)
    return None


def _extract_scores(stdout: str) -> Dict[str, float]:
    """从 run_batch.py 的 stdout 抓最后一行 JSON。"""
    for line in reversed(stdout.strip().splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except ValueError:
            continue
        if isinstance(data, dict):
            out: Dict[str, float] = {}
            for k, v in data.items():
                if isinstance(v, dict):        # {vid: {metric: val}}
                    for val in v.values():
                        if isinstance(val, (int, float)):
                            out[k] = float(val)
                            break
                elif isinstance(v, (int, float)):
                    out[k] = float(v)
            return out
    return {}


def score_variants(model: str, wt_seq: str, variants: Iterable[Any],
                   remote_cfg: Dict[str, Any]) -> Dict[str, float]:
    """通过腾讯云 TAT 在 GPU 实例上真实打分, 返回 {variant_id: float}。

    参数:
      model: esm1v/saprot/venusrem/promep/evmutation。
      wt_seq: WT 序列。
      variants: 可迭代 Variant 或 dict(取 site/from/to 拼 F205G)。
      remote_cfg: 见模块 docstring。
    """
    try:
        import tencentcloud  # noqa: F401
    except ImportError as e:
        raise RuntimeError(_TAT_HINT) from e

    _patch_dns()
    client, models = _tat_client(remote_cfg)

    muts = []
    for v in variants:
        if isinstance(v, dict):
            muts.append(f"{v['from']}{v['site']}{v['to']}")
        else:
            muts.append(f"{getattr(v, 'from_aa')}{getattr(v, 'site')}{getattr(v, 'to_aa')}")
    if not muts:
        return {}

    work_dir = str(remote_cfg.get("work_dir", "/work") or "/work")
    flags = []
    if remote_cfg.get("models_dir"):
        flags.append(f"--models_dir {remote_cfg['models_dir']}")
    if remote_cfg.get("model_path"):
        flags.append(f"--model_path '{remote_cfg['model_path']}'")
    if model == "saprot" and remote_cfg.get("structure_path"):
        flags.append(f"--structure_path '{remote_cfg['structure_path']}'")
    if model == "saprot" and remote_cfg.get("chain"):
        flags.append(f"--chain {remote_cfg['chain']}")
    device = str(remote_cfg.get("device", "cuda") or "cuda")
    timeout = int(remote_cfg.get("timeout", 1800) or 1800)

    cmd = (f"cd {work_dir} && python3 run_batch.py --model {model} "
           f"--wt_seq '{wt_seq}' --mutations {','.join(muts)} "
           f"--device {device} " + " ".join(flags))
    LOG.info("远程打分: %s (%d 突变)", model, len(muts))
    inv_id = _run_command(client, models, remote_cfg, cmd, timeout)
    stdout = _wait_result(client, models, inv_id, timeout)
    if stdout is None:
        raise RuntimeError(f"远程打分未返回结果(model={model}); 检查实例 TAT 状态与 run_batch.py 日志")
    scores = _extract_scores(stdout)
    if not scores:
        LOG.warning("远程打分结果为空(model=%s); stdout 尾部: %r", model, stdout[-300:])
    return scores


__all__ = ["score_variants"]
