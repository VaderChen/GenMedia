"""Independent scalar FP64 oracle: no MLX or model runtime dependency."""
import json
import math
from pathlib import Path

cfg = dict(hidden_size=8, num_hidden_layers=2, num_attention_heads=2,
    intermediate_size=12, head_dim=4, global_head_dim=8,
    num_key_value_heads=1, num_global_key_value_heads=1, vocab_size=32,
    hidden_size_per_layer_input=0, num_kv_shared_layers=0, rms_norm_eps=1e-6,
    attention_k_eq_v=True, enable_moe_block=False, use_double_wide_mlp=False,
    layer_types=['sliding_attention', 'full_attention'],
    rope_parameters={'sliding_attention': dict(rope_theta=10000),
                     'full_attention': dict(rope_theta=1000000, partial_rotary_factor=.25)})
weights = {}
def add(name, shape, base=0, scale=.15):
    values=[base + scale * math.sin((i+1)*.73 + len(name)) for i in range(math.prod(shape))]
    weights[name]=dict(shape=shape, values=values)
def array(name):return weights[name]['values']
add('embed_tokens.weight', [32,8]); add('norm.weight',[8],.9,.1)
for i,dim in enumerate([4,8]):
    p=f'layers.{i}'
    for n in ['input_layernorm','post_attention_layernorm','pre_feedforward_layernorm','post_feedforward_layernorm']:
        add(f'{p}.{n}.weight',[8],.8,.2)
    add(p+'.layer_scalar',[1],.9,.05)
    for n in ['q_norm','k_norm']:add(f'{p}.self_attn.{n}.weight',[dim],.7,.15)
    for n,o,inp in [('q_proj',2*dim,8),('k_proj',dim,8),('o_proj',8,2*dim)]+([('v_proj',dim,8)] if i==0 else []):
        add(f'{p}.self_attn.{n}.weight',[o,inp])
    for n,o,inp in [('gate_proj',12,8),('up_proj',12,8),('down_proj',8,12)]:add(f'{p}.mlp.{n}.weight',[o,inp])
def linear(x,name):
    w=array(name+'.weight');o,inp=weights[name+'.weight']['shape']
    return [[sum(row[j]*w[k*inp+j] for j in range(inp)) for k in range(o)] for row in x]
def norm(x,name=None):
    return [[v/math.sqrt(sum(t*t for t in row)/len(row)+1e-6)*(array(name+'.weight')[j] if name else 1)
             for j,v in enumerate(row)] for row in x]
def rotate(x,pos,dim,part,base):
    y=x.copy(); half=dim//2
    for j in range(int(half*part)):
        a=pos*base**(-2*j/dim);c,s=math.cos(a),math.sin(a)
        y[j]=x[j]*c-x[half+j]*s;y[half+j]=x[half+j]*c+x[j]*s
    return y
def feed(x,i):
    p=f'layers.{i}';dim=[4,8][i];h=norm(x,p+'.input_layernorm')
    q0=linear(h,p+'.self_attn.q_proj');k0=linear(h,p+'.self_attn.k_proj')
    v0=linear(h,p+'.self_attn.v_proj') if i==0 else k0
    k=norm(k0,p+'.self_attn.k_norm');v=norm(v0)
    part,base=(1,10000) if i==0 else (.25,1000000)
    k=[rotate(row,t,dim,part,base) for t,row in enumerate(k)]
    att=[]
    for t,row in enumerate(q0):
        a=[]
        for head in range(2):
            q=rotate(norm([row[head*dim:(head+1)*dim]],p+'.self_attn.q_norm')[0],t,dim,part,base)
            logits=[sum(q[j]*k[s][j] for j in range(dim)) for s in range(t+1)]
            probabilities=[math.exp(z-max(logits)) for z in logits];total=sum(probabilities)
            a += [sum(probabilities[s]/total*v[s][j] for s in range(t+1)) for j in range(dim)]
        att.append(a)
    att=norm(linear(att,p+'.self_attn.o_proj'),p+'.post_attention_layernorm')
    x=[[a+b for a,b in zip(r,s)] for r,s in zip(x,att)]
    h=norm(x,p+'.pre_feedforward_layernorm');gate=linear(h,p+'.mlp.gate_proj');up=linear(h,p+'.mlp.up_proj')
    h=[[.5*a*(1+math.tanh(math.sqrt(2/math.pi)*(a+.044715*a**3)))*b for a,b in zip(r,s)] for r,s in zip(gate,up)]
    h=norm(linear(h,p+'.mlp.down_proj'),p+'.post_feedforward_layernorm')
    return [[(a+b)*array(p+'.layer_scalar')[0] for a,b in zip(r,s)] for r,s in zip(x,h)]
tokens=[2,5,7];hidden=[[v*math.sqrt(8) for v in array('embed_tokens.weight')[t*8:(t+1)*8]] for t in tokens]
states=[hidden]
for i in range(2):hidden=feed(hidden,i);states.append(hidden)
states[-1]=norm(states[-1],'norm')
out=dict(configuration={'text_config':cfg},tokens=tokens,
         weights={'model.language_model.'+k:v for k,v in weights.items()},
         states=[[v for row in state for v in row] for state in states])
Path(__file__).with_name('gemma4-tiny.json').write_text(json.dumps(out,indent=2)+'\n')
