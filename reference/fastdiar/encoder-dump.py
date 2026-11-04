"""Dump pinned official streaming encoder stages, histories and whole-waveform embeddings.

Usage: uv run encoder-dump.py small|large --source OFFICIAL --checkpoint MODEL.pt --out DIR [--audio input.wav]
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from pins import COMMIT,MODELS,load

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('model',choices=MODELS)
for key in ('source','checkpoint','out'):p.add_argument('--'+key,type=Path,required=True)
p.add_argument('--audio',type=Path)
p.add_argument('--whole',action='store_true',help='save the whole audio and final embeddings without intermediate stage dumps')
a=p.parse_args();m=load(a.source,a.checkpoint,a.model)
from fastdiar.encoder import StreamingEncoder,StreamingInference

def array(x):return x.detach().float().cpu().numpy()
def save(path,x):np.save(path,np.ascontiguousarray(x,dtype=np.float32))
def cache_name(key):
    if key=='stem.conv':return 'backbone.stem.0'
    stage,block,conv=key.split('.')
    return f'backbone.stages.{stage[5:]}.conv_blocks.{block[5:]}.conv_block.{conv}'

rng=np.random.default_rng(20261103)
cases={'noise':rng.normal(0,.04,5120*3+977).astype(np.float32),'silence':np.zeros(5120+1280+251,dtype=np.float32),
       'tiny':np.zeros(1200,dtype=np.float32)}
if a.audio:
    import soundfile as sf
    samples,rate=sf.read(a.audio,dtype='float32');assert rate==16000 and samples.ndim==1
    cases['speech']=samples if a.whole else samples[:5120*4+1711]
if a.whole:
    if not a.audio:raise SystemExit('--whole requires --audio')
    cases={'speech':cases['speech']}
for name,audio in cases.items():
    root=a.out/name;root.mkdir(parents=True,exist_ok=True);save(root/'audio.npy',audio)
    streamer=StreamingEncoder(m,shift_frames=4)
    embeddings=[]
    for pos in range(0,len(audio),5120):embeddings.extend(streamer.push(torch.from_numpy(audio[pos:pos+5120])))
    embeddings.extend(streamer.flush())
    save(root/'expected.npy',np.stack([array(e) for e in embeddings]) if embeddings else np.zeros((0,m.linear.out_features),np.float32))
    engine=StreamingInference(m);steps=[]
    if a.whole:
        (root/'meta.json').write_text(json.dumps({'model':a.model,'officialCommit':COMMIT,'checkpointSha256':MODELS[a.model], 'audioFloat32Sha256':hashlib.sha256(audio.tobytes()).hexdigest(),'sampleRate':16000,'samples':len(audio),'stepSamples':5120,'torch':torch.__version__},indent=2)+'\n')
        print(name,len(embeddings),'embedding frames',flush=True)
        continue
    def spec_wrapper(original):
        def run(x):
            y=original(x);save(step/'features.npy',array(y)[0].T);return y
        return run
    engine._compute_spec_streaming=spec_wrapper(engine._compute_spec_streaming)
    original_stage=engine._stage_streaming
    def run_stage(stage,outputs,index):
        for j,x in enumerate(outputs):save(step/f'stage{index}-input{j}.npy',array(x)[0].T)
        y=original_stage(stage,outputs,index);save(step/f'stage{index}.npy',array(y)[0].T);return y
    engine._stage_streaming=run_stage
    stem_hook=m.backbone.stem[2].register_forward_hook(lambda module,inputs,output:save(step/'stem.npy',array(output)[0].T))
    original_backbone=engine._backbone_streaming
    def run_backbone(x):
        y=original_backbone(x);save(step/'backbone.npy',array(y).reshape(1,-1,y.shape[-1])[0].T);return y
    engine._backbone_streaming=run_backbone
    for pos in range(0,len(audio),5120):
        n=min(5120,len(audio)-pos);n=n//engine.min_chunk_samples*engine.min_chunk_samples
        if not n:continue
        step=root/f'step{len(steps)}';step.mkdir(exist_ok=True);save(step/'audio.npy',audio[pos:pos+n])
        previous={cache_name(k):array(v)[0].transpose(0,2,1) for k,v in engine.state.conv_buffers.items()}
        for key,value in previous.items():save(step/f'input-{key}.npy',value)
        result=engine.process_chunk(torch.from_numpy(audio[pos:pos+n]));save(step/'embeddings.npy',array(result)[0])
        caches={cache_name(k):array(v)[0].transpose(0,2,1) for k,v in engine.state.conv_buffers.items()}
        for key,value in caches.items():save(step/f'output-{key}.npy',value)
        steps.append({'samples':n,'inputCaches':list(previous),'outputCaches':list(caches)})
    stem_hook.remove()
    (root/'meta.json').write_text(json.dumps({'model':a.model,'officialCommit':COMMIT,'checkpointSha256':MODELS[a.model],'steps':steps,'sampleRate':16000,'samples':len(audio),'stepSamples':5120,'torch':torch.__version__,'seed':20261103,'audioFloat32Sha256':hashlib.sha256(audio.tobytes()).hexdigest()},indent=2)+'\n')
    print(name,len(steps),len(embeddings),'embedding frames',flush=True)
