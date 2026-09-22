# Independent NumPy dense-attention oracle; tiny tensors only, never a production runtime.
import numpy as np,json,pathlib,math
p=pathlib.Path(__file__).resolve().parents[2] / 'Tests/QwenImage21RuntimeTests/Fixtures';p.mkdir(exist_ok=True)
rng=np.random.default_rng(21);d=8
weights={}
def dense(name,out,inn): weights[name+'.weight']=(rng.standard_normal((out,inn))*.12).astype(np.float32)
for n,o,i in [('img_in',d,64),('txt_in.in_layer',d,d),('txt_in.out_layer',d,d),('modulation.0',d*4,d),('norm_out.linear',d,d),('proj_out',64,d),('time_text_embed.linear_1',d,256),('time_text_embed.linear_2',d,d)]:dense(n,o,i)
weights['txt_in.text_norm.weight']=(rng.standard_normal(d)*.1).astype(np.float32)
for n in ['to_q','to_k','to_v','to_out.0']:dense('transformer_blocks.0.attn.'+n,d,d)
for n in ['norm_q','norm_k']:weights['transformer_blocks.0.attn.'+n+'.weight']=np.ones(d,np.float32)
for n,o,i in [('proj',24,d),('gate_layer',24,d),('out',d,24)]:dense('transformer_blocks.0.img_mlp.'+n,o,i)
def lin(x,n):return x@weights[n+'.weight'].T
def silu(x):return x/(1+np.exp(-x))
def rms(x,w):return x/np.sqrt(np.mean(x*x,axis=-1,keepdims=True)+1e-6)*w
def ln(x):return (x-x.mean(-1,keepdims=True))/np.sqrt(x.var(-1,keepdims=True)+1e-6)
def gelu(x):return .5*x*(1+np.tanh(np.sqrt(2/np.pi)*(x+.044715*x**3)))
def run(edit):
 slots=[False,True,False,False] if edit else [False]*4
 c=rng.standard_normal((1,4,d)).astype(np.float32)
 z=rng.standard_normal((1,8 if edit else 4,64)).astype(np.float32)
 text=lin(gelu(lin(rms(c,weights['txt_in.text_norm.weight']+1),'txt_in.in_layer')),'txt_in.out_layer');im=lin(z,'img_in')
 vals=[];pos=[];imageids=[];position=0;imoffset=0
 for i,slot in enumerate(slots):
  if slot:
   vals.extend(im[0,imoffset:imoffset+4]);imoffset+=4
   pos.extend([[position,y-1,x-1] for y in range(2) for x in range(2)]);position+=2;imageids.extend([0]*4)
  else:vals.append(text[0,i]);pos.append([position]*3);position+=1;imageids.append(-1)
 target=len(vals);vals.extend(im[0,imoffset:]);pos.extend([[position,y-1,x-1] for y in range(2) for x in range(2)]);imageids.extend([1]*4)
 x=np.array(vals)[None];n=len(vals)
 t=np.array([700.,0.],np.float32)[:,None]*np.exp(-np.log(10000.)*np.arange(128)/128)
 temb=lin(silu(lin(np.concatenate([np.cos(t),np.sin(t)],-1),'time_text_embed.linear_1')),'time_text_embed.linear_2')
 mod=lin(silu(temb),'modulation.0');s1,g1,s2,g2=np.split(mod,4,-1)
 def rows(v):return np.concatenate([np.repeat(v[1:2],target,0),np.repeat(v[:1],n-target,0)],0)[None]
 a=ln(x)*(1+rows(s1));pfx='transformer_blocks.0'
 angles=np.concatenate([np.array(pos)[:,axis,None]/10000**(np.arange(0,dim,2)/dim) for axis,dim in enumerate([2,2,4])],-1)
 def rope(v):
  a=v.reshape(1,n,4,2);re=a[...,0];im=a[...,1];return np.stack([re*np.cos(angles)-im*np.sin(angles),re*np.sin(angles)+im*np.cos(angles)],-1).reshape(1,n,d)
 q=rope(rms(lin(a,pfx+'.attn.to_q'),np.ones(d)));k=rope(rms(lin(a,pfx+'.attn.to_k'),np.ones(d)));v=lin(a,pfx+'.attn.to_v')
 scores=q@k.transpose(0,2,1)/np.sqrt(d)
 ids=np.array(imageids);allowed=(np.arange(n)[:,None]>=np.arange(n)[None,:])|((ids[:,None]==ids[None,:])&(ids[:,None]>=0))
 scores=np.where(allowed,scores,-np.inf);probs=np.exp(scores-scores.max(-1,keepdims=True));probs/=probs.sum(-1,keepdims=True)
 x=x+np.tanh(rows(g1))*lin(probs@v,pfx+'.attn.to_out.0');a=ln(x)*(1+rows(s2))
 x=x+np.tanh(rows(g2))*lin(silu(lin(a,pfx+'.img_mlp.gate_layer'))*lin(a,pfx+'.img_mlp.proj'),pfx+'.img_mlp.out')
 out=lin(ln(x[:,target:])*(1+lin(silu(temb[:1]),'norm_out.linear')[:,None]),'proj_out')
 return {'slots':slots,'conditioning':c.flatten().tolist(),'latents':z.flatten().tolist(),'expected':out.flatten().tolist()}
fixture={'weights':{k:{'shape':list(v.shape),'values':v.flatten().tolist()} for k,v in weights.items()},'cases':[run(False),run(True)]}
(p/'transformer-oracle.json').write_text(json.dumps(fixture,separators=(',',':'))+'\n')
