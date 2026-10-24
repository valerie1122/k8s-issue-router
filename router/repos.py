"""Per-repository settings. Everything repo-specific lives here; the pipeline code is shared."""
import json

import yaml

BOTS = {"k8s-ci-robot", "kubernetes-prow", "k8s-triage-robot", "github-actions", "pytorch-bot",
        "pytorchmergebot", "facebook-github-bot", "pytorch-bot[bot]", "github-actions[bot]"}

REPOS = {
    "k8s": {
        "owner": "kubernetes", "name": "kubernetes",
        "label_prefix": "sig/",           # the routing label, e.g. sig/node
        "team_word": "SIG",
        "issues": "data/interim/issues_3y_orig.jsonl",
        "dataset": "data/processed/dataset.jsonl",
        "splits": [("train", "2000-01-01", "2026-04-01"), ("val", "2026-04-01", "2026-07-01"),
                   ("test", "2026-07-01", "2100-01-01")],
        "none_class": False,              # no SIG after 30 days = unknown, not a class
    },
    "pytorch": {
        "owner": "pytorch", "name": "pytorch",
        "label_prefix": "oncall: ",       # the routing label, e.g. "oncall: pt2"
        "route_prefixes": ("oncall: ", "module:"),  # any of these = someone made the routing decision
        "team_word": "oncall team",
        "issues": "data/interim/pytorch_issues_1y_orig.jsonl",
        "dataset": "data/processed/pytorch_dataset.jsonl",
        "splits": [("train", "2000-01-01", "2026-06-01"), ("val", "2026-06-01", "2026-08-01"),
                   ("test", "2026-08-01", "2100-01-01")],
        "none_class": True,               # triaged with no oncall label = routed to a module team, a real answer
    },
}

# What each pytorch oncall queue covers. GitHub's own label descriptions are mostly empty, so this is
# written from the label names/descriptions plus the PyTorch component layout.
PYTORCH_TEAMS = {
    "pt2": "torch.compile stack: TorchDynamo graph capture, TorchInductor codegen (GPU/Triton), AOTAutograd, "
           "AOTInductor, functorch transforms used by compile, dynamic shapes, recompilations, graph breaks.",
    "export": "torch.export and exported programs, ExportedProgram serialization, non-strict export.",
    "distributed": "torch.distributed core: c10d process groups, NCCL/Gloo backends, DDP, collectives, RPC.",
    "distributed infra": "distributed infrastructure: launcher (torchrun/elastic), rendezvous, store, distributed tests infra.",
    "distributed parallelisms": "FSDP/FSDP2, tensor/pipeline/context parallel, DTensor, DeviceMesh.",
    "distributed checkpointing": "torch.distributed.checkpoint (DCP), sharded state dicts.",
    "cpu inductor": "TorchInductor on CPU (C++/OpenMP codegen, x86 vectorization), triaged by the Intel team.",
    "profiler": "torch.profiler, Kineto, CPU/GPU traces, memory profiling.",
    "jit": "TorchScript: torch.jit.script/trace, the JIT runtime.",
    "quantization": "quantization APIs (torch.ao.quantization, PT2E quantization), quantized kernels.",
    "releng": "CI and release engineering: wheels, binaries, nightly builds, packaging.",
    "mobile": "mobile: iOS/Android, ExecuTorch-facing mobile runtime issues.",
    "visualization": "TensorBoard / visualization utilities.",
    "fx": "torch.fx tracing and graph transformations (outside of torch.compile).",
    "r2p": "R2P (elastic) oncall.",
    "onnx": "ONNX export.",
}
PYTORCH_NONE = ("none: no oncall queue; the issue is triaged by a module owner instead (CUDA, ROCm, XPU, MPS, CI, "
                "nn modules, autograd, optimizers, data loading, docs, build, correctness of eager kernels, ...).")


def teams(repo):
    """[(label, description)] the models may answer with (excluding none)."""
    if repo == "k8s":
        sigs = yaml.safe_load(open("data/raw/sigs.yaml"))["sigs"]
        return [(s["label"], " ".join((s.get("mission_statement") or "").split())[:600]
                 + " Subprojects: " + ", ".join(p["name"] for p in (s.get("subprojects") or []))[:400])
                for s in sigs]
    return list(PYTORCH_TEAMS.items())


def catalog(repo):
    lines = [f"- {k}: {v}" for k, v in teams(repo)]
    if REPOS[repo]["none_class"]:
        lines.append(f"- {PYTORCH_NONE}")
    return "\n".join(lines)
