"""Pure helpers for Section 8. No final-holdout access."""
import numpy as np
import pandas as pd

def invert_difference8(last_levels, differences, lag):
    history=list(np.asarray(last_levels,float))
    if len(history)<lag: raise ValueError('Insufficient inverse-transform anchors.')
    out=[]
    for difference in np.asarray(differences,float):
        level=history[-lag]+difference
        history.append(level);out.append(level)
    return np.asarray(out)

def starts8(dates, lookback=56, horizon=48, stride=7):
    dates=np.asarray(dates,dtype='datetime64[D]')
    found=[]
    for pos in range(lookback,len(dates)-horizon+1,stride):
        z=dates[pos-lookback:pos+horizon]
        if np.all(np.diff(z)==np.timedelta64(1,'D')): found.append(pos)
    return found

def context8(frame, scaler):
    # Open-with-zero observations are unknown/suspect, not fabricated valid sales.
    valid=((frame.Open.eq(0)&frame.Sales.eq(0)) | (frame.Open.eq(1)&frame.Sales.gt(0)))
    z=scaler.transform(np.log1p(frame.Sales.to_numpy(float)).reshape(-1,1)).ravel()
    z[~valid.to_numpy()]=0.0
    return np.column_stack([z,frame.Open.to_numpy(float),valid.to_numpy(float)]).astype('float32')

def masked_mse_numpy8(pred,y,mask):
    if np.asarray(mask).sum()==0: raise ValueError('No valid targets.')
    return float((((np.asarray(pred)-np.asarray(y))**2)*mask).sum()/np.asarray(mask).sum())
