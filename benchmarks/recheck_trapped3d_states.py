"""Memory-bounded NumPy/SciPy re-evaluation, independent of becgpp imports.

Full free-space 4x4x4-cell-averaged Newton convolution for N<=128; local and
spectral terms for all four grids. This is a discrete-state audit, not a
separate optimizer or a proof of continuum/global accuracy.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
import numpy as np
from scipy.fft import fftn, ifftn, rfftn, irfftn

DEFAULT_RAW=Path('data/raw/trapped3d_release_2026_10_31')


def calculate(p_path,N,L,full=True):
    p=np.load(p_path,mmap_mode='r',allow_pickle=False)
    assert p.shape==(N,)*3 and p.dtype==np.complex128
    h=2*L/N;dv=h**3;x=(np.arange(N)-N//2)*h
    mass=trap=local=0.
    rho=np.empty((N,N,N),dtype=np.float64)
    yz=x[:,None]**2+x[None,:]**2
    for i in range(N):
        r=p[i].real**2+p[i].imag**2;rho[i]=r
        mass+=r.sum()*dv;trap+=(.5*(x[i]**2+yz)*r).sum()*dv;local+=50*(r*r).sum()*dv
    f=fftn(p,workers=1)
    k=2*np.pi*np.fft.fftfreq(N,d=h);kyz=k[:,None]**2+k[None,:]**2
    kinetic=0.
    for i in range(N):kinetic+=.5*np.sum((k[i]**2+kyz)*np.abs(f[i])**2)*dv/N**3
    result=dict(N=N,L=L,mass=float(mass),Ekin=float(kinetic),Etrap=float(trap),Econtact=float(local),full_newton=full)
    if not full:return result
    M=2*N
    d=np.arange(M,dtype=np.float64);d=np.where(d<=M//2,d,d-M)*h
    nodes,weights=np.polynomial.legendre.leggauss(4)
    ker=np.empty((M,M,M),dtype=np.float64)
    # Build only four x-planes at once, avoiding full coordinate meshgrids.
    for first in range(0,M,4):
        acc=np.zeros((min(4,M-first),M,M),dtype=np.float64)
        for u,wu in zip(nodes,weights):
            xx=(d[first:first+4,None,None]+h*u/2)**2
            for v,wv in zip(nodes,weights):
                xy=xx+(d[None,:,None]+h*v/2)**2
                for w,ww in zip(nodes,weights):
                    acc+=(wu*wv*ww/8)/np.sqrt(xy+(d[None,None,:]+h*w/2)**2)
        ker[first:first+4]=acc
    ker[0,0,0]=2.3800773640/h
    kernel_fft=rfftn(ker,workers=1);del ker
    padded=np.zeros((M,)*3,dtype=np.float64)
    start=N//2;sl=(slice(start,start+N),)*3;padded[sl]=rho
    rf=rfftn(padded,workers=1);del padded
    rf*=kernel_fft;del kernel_fft
    conv=irfftn(rf,s=(M,)*3,workers=1);del rf
    phi=(-20*dv*conv[sl]).copy();del conv
    Eg=float(.5*np.sum(rho*phi)*dv)
    E=kinetic+trap+local+Eg
    muE=E+local+Eg
    for i in range(N):f[i]*=.5*(k[i]**2+kyz)
    Hp=ifftn(f,workers=1);del f
    for i in range(N):Hp[i]+=(.5*(x[i]**2+yz)+100*rho[i]+phi[i])*p[i]
    muH=float(np.sum((np.conj(p)*Hp).real)*dv)
    resid=math.sqrt(float(np.sum(np.abs(Hp-muH*p)**2)*dv))/max(1,abs(muH))
    virial=2*kinetic-2*trap+3*local+Eg
    vscale=max(1,abs(kinetic)+abs(trap)+abs(local)+2*abs(Eg))
    result.update(Egrav=Eg,energy=float(E),mu_H=muH,mu_E=float(muE),residual=resid,
                  virial_relative=float(abs(virial)/vscale),phi_center=float(phi[N//2,N//2,N//2]))
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--N',type=int,required=True);parser.add_argument('--state');parser.add_argument('--L',type=float,default=8.)
    parser.add_argument('--local-only',action='store_true');parser.add_argument('--label')
    parser.add_argument('--raw',type=Path,default=DEFAULT_RAW)
    parser.add_argument('--out',type=Path,default=Path('state_audit'))
    args=parser.parse_args(); RAW=args.raw; OUT=args.out; OUT.mkdir(parents=True,exist_ok=True)
    recorded=None
    if args.state:p=Path(args.state)
    else:
        recorded=json.loads((RAW/f'jobs/grid_N{args.N}/result.json').read_text())
        p=RAW/recorded['state_path']
    begin=time.perf_counter();result=calculate(p,args.N,args.L,not args.local_only)
    result['seconds']=time.perf_counter()-begin
    if recorded:
        errors={k:abs(result[k]-recorded['energy_components'][k]) for k in ('Ekin','Etrap','Econtact')}
        assert max(errors.values())<2e-12
        result['component_absolute_errors']=errors
        if not args.local_only:
            result['scaled_energy_error']=abs(result['energy']-recorded['energy'])/max(1,abs(recorded['energy']))
            result['residual_absolute_error']=abs(result['residual']-recorded['residual'])
            assert result['scaled_energy_error']<2e-12 and result['residual_absolute_error']<2e-12
    label=args.label or ('state_N'+str(args.N))
    result['scope']='Independent NumPy/SciPy discrete-state evaluation; no becgpp imported, no optimization performed'
    (OUT/(label+'.json')).write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
