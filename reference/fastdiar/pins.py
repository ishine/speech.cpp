"""Load the pinned official streaming encoder and verify every checkpoint before conversion or dumping."""
import hashlib
import subprocess
import sys
from pathlib import Path

COMMIT = "f48bff53b9a6a0cce4539e94b6def95f609eaea4"
MODELS = {
    "small": "90be5da5e86645c39eceaedfd9f05cc7d35734d9105691da3e69b3e4ec47fb2a",
    "large": "f6d71444eb513654a4b8c6563d11196d19283759884d4986488af3a57361228d",
}


def load(source, checkpoint, name):
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if revision != COMMIT or subprocess.run(["git", "-C", str(source), "diff", "--quiet", "HEAD", "--", "fastdiar"]).returncode:
        raise SystemExit(f"official source must be clean at {COMMIT}")
    digest = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
    if digest != MODELS[name]:
        raise SystemExit(f"checkpoint SHA-256 is {digest}, not {MODELS[name]}")
    sys.path.insert(0, str(Path(source).resolve()))
    import torch
    from fastdiar.encoder import load_streaming_model
    torch.set_grad_enabled(False)
    torch.set_num_threads(1)
    return load_streaming_model(str(checkpoint), device="cpu", dtype=torch.float32)
