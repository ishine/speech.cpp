"""Convert a pinned FASTDIAR streaming encoder to layout 2 F32 GGUF.

Usage: uv run convert.py small|large --source OFFICIAL --checkpoint MODEL.pt --silero SILERO.gguf --out MODEL.gguf
Grouped downsampling weights are expanded with exact zeros; depthwise residual kernels stay compact.
BatchNorm training counters are not read in evaluation and are the only omitted state tensors.
The keys and tensors of Silero VAD's GGUF, which the official diarizer runs, are carried unchanged: their names do not collide.
"""
import argparse
from pathlib import Path
import numpy as np
import torch
from gguf import GGUFReader, GGUFValueType, GGUFWriter
from pins import COMMIT, MODELS, load

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("model", choices=MODELS)
p.add_argument("--source", type=Path, required=True)
p.add_argument("--checkpoint", type=Path, required=True)
p.add_argument("--silero", type=Path, required=True)
p.add_argument("--out", type=Path, required=True)
a = p.parse_args()
m = load(a.source, a.checkpoint, a.model)
from fastdiar.encoder import StreamingInference
s = StreamingInference(m)
a.out.parent.mkdir(parents=True, exist_ok=True)
w = GGUFWriter(str(a.out), "fastdiar")
for key, value in {
    "general.name": "FASTDIAR-" + a.model, "general.organization": "Microsoft", "general.basename": "FASTDIAR",
    "general.size_label": "2.4M" if a.model == "small" else "10.4M", "general.license": "mit",
    "general.source.repo_url": "https://github.com/microsoft/fastdiar",
    "general.source.url": "https://github.com/microsoft/fastdiar/tree/" + COMMIT,
    "speech.requires": "0.10.2", "speech.task": "diarization",
}.items(): w.add_string(key, value)
w.add_uint32("general.file_type", 0)
w.add_uint32("speech.layout", 2)
w.add_uint32("speech.sample_rate", 16000)
for key,value in {"mels":s._stft_melbanks.shape[0],
                  "channels":m.backbone.stem[0].out_channels,"stage_count":len(m.backbone.stages),
                  "hop":s.hop_length,"window":s._stft_real_kernel.shape[-1],"bins":s._stft_real_kernel.shape[0],
                  "time_stride":s.time_stride,"embedding_dim":m.linear.out_features,
                  "head_channels":m.backbone.head.out_channels if isinstance(m.backbone.head,torch.nn.Conv2d) else 0}.items():
    w.add_uint32("fastdiar." + key, int(value))
w.add_bool("fastdiar.time_upsampling",m.time_upsampling)
for key,value in {"norm_eps":s._norm_eps,"stft_eps":s._stft_eps,"cms_eps":s._cms_eps,"variance_floor":1e-12,
                  "normalize_eps":1e-12,"layer_eps":m.backbone.stem[1].eps,"bn_eps":m.bn.eps}.items():
    w.add_float32("fastdiar." + key,float(value))
for i,stage in enumerate(m.backbone.stages):
    prefix=f"fastdiar.stage.{i}."
    for key,value in {"channels":stage.to2d.c,"mels":stage.to2d.f,"mid_channels":stage.downsample.out_channels,
                      "freq_stride":stage.downsample.stride[0],"time_stride":stage.downsample.stride[1],
                      "groups":stage.downsample.groups,"blocks":len(stage.conv_blocks),
                      "project":int(isinstance(stage.project,torch.nn.Sequential)),
                      "bottleneck":stage.context.red_dim_conv[0].out_channels}.items():w.add_uint32(prefix+key,int(value))
    w.add_float32(prefix+"bn_eps",float(stage.conv_blocks[0].conv_block.bn1.eps))
    if isinstance(stage.project,torch.nn.Sequential):w.add_float32(prefix+"project_eps",float(stage.project[1].eps))
modules=dict(m.named_modules())
for name, tensor in m.state_dict().items():
    if name.endswith("num_batches_tracked"):continue
    data=tensor.detach().float().cpu().numpy()
    module=modules.get(name.rsplit('.',1)[0])
    if isinstance(module,torch.nn.Conv2d) and name.endswith('.weight'):
        depthwise='conv_blocks' in name and name.rsplit('.',2)[-2] in ('conv1','conv2')
        if module.groups>1 and not depthwise:
            expanded=np.zeros((module.out_channels,module.in_channels,*module.kernel_size),dtype=np.float32)
            ci=module.in_channels//module.groups;co=module.out_channels//module.groups
            for g in range(module.groups):expanded[g*co:(g+1)*co,g*ci:(g+1)*ci]=data[g*co:(g+1)*co]
            data=expanded
        data=data.transpose(0,1,3,2)
    elif isinstance(module,torch.nn.Conv1d) and name.endswith('.weight'):
        assert module.kernel_size==(1,)
        data=data[:,:,0]
    elif name.endswith('.aggregate.w') or name=='backbone.fin_wght1d.w':
        data=data[0,:,:,0].T
    elif name.startswith('spec.'):
        data=tensor.numpy().reshape(2) if name.endswith('flipped_filter') else data.squeeze(1) if 'kernel_pt' in name else data.squeeze(-1)
    w.add_tensor(name,np.ascontiguousarray(data,dtype=np.float32))
silero = GGUFReader(a.silero)
for key, field in silero.fields.items():
    if key.startswith("silero-vad."):
        w.add_key_value(key, field.contents(), field.types[0], sub_type=field.types[-1] if field.types[0] == GGUFValueType.ARRAY else None)
for tensor in silero.tensors:
    w.add_tensor(tensor.name, np.array(tensor.data, dtype=np.float32))
w.write_header_to_file();w.write_kv_data_to_file();w.write_tensors_to_file();w.close()
print(a.out)
