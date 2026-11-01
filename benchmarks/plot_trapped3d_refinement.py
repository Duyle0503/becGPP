"""Regenerate publication figure from the immutable release refinement data."""
import argparse
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--raw',type=Path,required=True)
parser.add_argument('--out',type=Path,required=True)
args=parser.parse_args(); RAW=args.raw; args.out.parent.mkdir(parents=True,exist_ok=True)
rows=[json.loads((RAW/f'jobs/grid_N{n}/result.json').read_text()) for n in [96,128,160,192]]
s=json.loads((RAW/'SUMMARY.json').read_text())
h=np.array([r['dx'] for r in rows]);e=np.array([r['energy'] for r in rows])
fit=s['metrics']['grid']['h2_fit']
plt.rcParams.update({'font.family':'serif','font.size':9,'mathtext.fontset':'cm','axes.linewidth':.75,
    'xtick.direction':'in','ytick.direction':'in','legend.frameon':False,'savefig.bbox':'tight'})
fig,ax=plt.subplots(1,2,figsize=(6.8,2.65),layout='constrained')
ax[0].plot(h*h,e,'o',color='#1b567a',ms=4,label='release 1.2.2')
xx=np.linspace(0,max(h*h)*1.03,120)
ax[0].plot(xx,fit['intercept']+fit['slope']*xx,'--',color='#b55332',lw=1.2,label='last-three-grid fit')
ax[0].plot([0],[fit['intercept']],'o',mfc='white',mec='#b55332',ms=4)
ax[0].set(xlabel=r'$\Delta x^2$',ylabel=r'Finite-grid energy $E$')
ax[0].legend(loc='upper left',fontsize=8)
ax[0].text(.98,.04,r'$p=1.979,\ 1.987$',transform=ax[0].transAxes,ha='right',fontsize=9)
v=np.array([r['virial_relative'] for r in rows])
ax[1].loglog(h,v,'o-',color='#1b567a',lw=1.2,ms=4,label='computed')
ax[1].loglog(h,v[-1]*(h/h[-1])**2,':',color='#b55332',lw=1.2,label=r'$\Delta x^2$ guide')
ax[1].axhline(5e-4,color='.4',linestyle='--',lw=.9,label='finest-grid criterion')
ax[1].set(xlabel=r'$\Delta x$',ylabel='Scaled virial residual')
ax[1].set_xticks(h[::-1], labels=['0.0833', '0.100', '0.125', '0.167'])
ax[1].xaxis.set_minor_locator(NullLocator())
ax[1].legend(fontsize=8,loc='lower right')
for a in ax:a.tick_params(top=True,right=True)
dest=args.out
fig.savefig(dest)
fig.savefig(dest.with_suffix('.png'),dpi=180)
plt.close(fig)
print(dest)
