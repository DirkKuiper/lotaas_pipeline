"""Fold/refine periodic candidates and retain the data behind review plots."""
from pathlib import Path
import os
import numpy as np


def folded_profile(values,dt,frequency,bins,subints=1):
    t=np.arange(len(values))*dt
    phase=np.floor(np.remainder(t*frequency,1.)*bins).astype(int)
    sub=np.minimum((np.arange(len(values))*subints)//len(values),subints-1)
    indices=sub*bins+phase
    counts=np.bincount(indices,minlength=subints*bins).reshape(subints,bins)
    sums=np.bincount(indices,weights=values,minlength=subints*bins).reshape(subints,bins)
    return sums,counts


def profile_chi2(sums,counts,variance):
    good=counts>0
    mean=sums.sum()/counts.sum()
    return float(np.sum((sums[good]-mean*counts[good])**2/counts[good])/max(variance,np.finfo(float).tiny))


def refine_fold(values,dt,candidate,config):
    bins=min(config['fold_bins'],max(8,int(candidate['period_seconds']/dt)))
    variance=float(np.var(values))
    centre=candidate['frequency_hz']
    radius=candidate['frequency_resolution_hz']/(2*candidate['harmonic_count'])
    frequencies=np.linspace(max(centre-radius,1/(len(values)*dt)),centre+radius,33)
    scores=[]
    for f in frequencies:
        sums,counts=folded_profile(values,dt,f,bins)
        scores.append(profile_chi2(sums,counts,variance))
    best=int(np.argmax(scores)); frequency=float(frequencies[best])
    sums,counts=folded_profile(values,dt,frequency,bins,config['fold_subintegrations'])
    profile=sums.sum(axis=0)/np.maximum(counts.sum(axis=0),1)
    profile-=np.mean(values)
    # Profile ordinate is a single-bin noise scale, not search significance.
    errors=np.sqrt(variance/np.maximum(counts.sum(axis=0),1))
    return {'frequency':frequency,'period':1/frequency,'chi2':scores[best],
            'bins':bins,'profile':profile,'errors':errors,'sums':sums,'counts':counts,
            'frequencies':frequencies,'scores':np.asarray(scores)}


def catalogue_matches(metadata):
    """Catalogue context only; a match never creates or vetoes a detection."""
    from psrqpy import QueryATNF
    from astropy.coordinates import SkyCoord
    from astropy import units as u
    info=metadata.get('observation_info',{})
    if not info.get('RA (J2000)') or not info.get('DEC (J2000)'):
        return [],'coordinates unavailable'
    beam=SkyCoord(info['RA (J2000)'],info['DEC (J2000)'],unit=(u.hourangle,u.deg))
    try:
        from lotaas_reprocessing.atnf import query_atnf
        query=query_atnf(factory=QueryATNF,params=['PSRJ','RAJ','DECJ','DM','P0','F0','F1','PEPOCH'],
                        coord1=info['RA (J2000)'],coord2=info['DEC (J2000)'],radius=1.,checkupdate=False)
        rows=[]
        for _,row in query.pandas.iterrows():
            if not np.isfinite(row.get('P0',np.nan)) or not np.isfinite(row.get('DM',np.nan)):
                continue
            position=SkyCoord(row.RAJ,row.DECJ,unit=(u.hourangle,u.deg))
            period=float(row.P0)
            epoch=metadata.get('tstart_mjd')
            if epoch and np.isfinite(row.get('F1',np.nan)) and np.isfinite(row.get('PEPOCH',np.nan)):
                f=float(row.F0)+float(row.F1)*(epoch-float(row.PEPOCH))*86400
                if f>0: period=1/f
            rows.append({'name':str(row.PSRJ),'dm':float(row.DM),'period_seconds':period,
                         'separation_deg':float(beam.separation(position).deg)})
        return rows,'available'
    except Exception as error:
        # No live catalogue is required to search, fold, retain or review data.
        return [],f'unavailable: {type(error).__name__}'


def match_catalogue(candidate,period,catalogue):
    matches=[]
    for row in catalogue:
        if abs(candidate['dm']-row['dm'])>max(1.,2*candidate.get('dm_step',0.)):
            continue
        ratio=period/row['period_seconds']
        relation=min([1.,2.,3.,4.,.5,1/3,.25],key=lambda h:abs(ratio-h)/h)
        # Topocentric vs barycentric Doppler, catalogue age and finite resolution.
        tolerance=max(.001,candidate['frequency_resolution_hz']*period)
        if abs(ratio/relation-1)<=tolerance:
            matches.append(dict(row,period_ratio=relation))
    return sorted(matches,key=lambda r:r['separation_deg'])


def fold_subbands(filename,dm,frequency,bins,valid_seconds,bad_channels=()):
    """Independent frequency-phase diagnostic from the original filterbank.

    Fold each channel at its dispersive arrival phase, using only the same
    reference-time interval as the searched trial. Subtract channel means in
    each chunk; divide by measured channel noise before accumulating bands.
    """
    from .filterbank import FilterbankFile
    fil=FilterbankFile(str(filename))
    bands=min(16,fil.nchans)
    sums=np.zeros((bands,bins)); counts=np.zeros_like(sums)
    frequencies=np.asarray(fil.frequencies)
    delays=dm*(frequencies**-2-frequencies.max()**-2)/2.41e-4
    try:
        for start in range(0,fil.nspec,8192):
            data=fil.get_spectra(start,min(start+8192,fil.nspec))
            t=(start+np.arange(len(data)))*fil.tsamp
            for channel in range(fil.nchans):
                if channel in bad_channels:
                    continue
                reference=t-delays[channel]
                valid=(reference>=0)&(reference<valid_seconds)
                values=data[valid,channel].astype(float)
                if len(values)<2 or not np.isfinite(values).all():continue
                values-=np.mean(values)
                scale=np.std(values)
                if scale==0:continue
                phase=np.floor(np.remainder(reference[valid]*frequency,1)*bins).astype(int)
                band=channel*bands//fil.nchans
                sums[band]+=np.bincount(phase,weights=values/scale,minlength=bins)
                counts[band]+=np.bincount(phase,minlength=bins)
    finally:fil.close()
    image=sums/np.sqrt(np.maximum(counts,1))
    if frequencies[0]>frequencies[-1]: image=image[::-1]
    return image,np.array([frequencies.min(),frequencies.max()])


def plot_fold(path,fold,candidate,metadata,dm_curve,subbands=None,band_limits=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig=plt.figure(figsize=(12,9),layout='constrained')
    grid=fig.add_gridspec(3,2,height_ratios=[1,1.35,.7])
    ax=fig.add_subplot(grid[0,0]); bins=fold['bins']; phase=(np.arange(bins)+.5)/bins
    ax.errorbar(phase,fold['profile']/fold['errors'],yerr=np.ones(bins),color='#174b73',lw=1)
    ax.set(xlabel='Pulse phase',ylabel='Profile / bin noise',title='Integrated folded profile')
    ax=fig.add_subplot(grid[0,1]); ax.plot((fold['frequencies']-candidate['frequency_hz'])*1e6,fold['scores'],color='#174b73')
    ax.set(xlabel='Frequency offset (µHz)',ylabel='Fold χ²',title='Local period refinement')
    ax=fig.add_subplot(grid[1,0]); sums=fold['sums'];counts=fold['counts']
    means=sums/np.maximum(counts,1);means-=np.sum(sums,axis=1)[:,None]/np.maximum(np.sum(counts,axis=1)[:,None],1)
    image=means*np.sqrt(counts)
    limit=np.percentile(np.abs(image),99)
    ax.imshow(image,aspect='auto',origin='lower',extent=[0,1,0,candidate['observation_seconds']/60],cmap='magma',vmin=-limit,vmax=limit)
    ax.set(xlabel='Pulse phase',ylabel='Time (minutes)',title='Persistence across the observation')
    ax=fig.add_subplot(grid[1,1])
    if subbands is not None:
        ax.imshow(subbands,aspect='auto',origin='lower',extent=[0,1,*band_limits],cmap='magma')
        ax.set(xlabel='Pulse phase',ylabel='Frequency (MHz)',title='Folded original filterbank')
    else:
        ax.scatter(dm_curve[:,0],dm_curve[:,1],s=14,color='#174b73') if len(dm_curve) else None
        ax.set(xlabel='DM (pc cm⁻³)',ylabel='Nominal −log₁₀ p',title='Search response near this period')
    ax=fig.add_subplot(grid[2,:]);ax.axis('off')
    matches=candidate.get('catalogue_matches',[])
    known=', '.join(r['name'] for r in matches) if matches else 'none within matching tolerances'
    text=(f"P = {fold['period']:.9f} s (topocentric)    DM = {candidate['dm']:.3f} pc cm⁻³\n"
          f"Search −log₁₀ p = {candidate['statistic']:.2f} (nominal)    Harmonics = {candidate['harmonic_count']}    "
          f"Valid duration = {candidate['observation_seconds']:.1f} s    Sampling = {candidate['effective_sampling_seconds']*1000:.3f} ms\n"
          f"Catalogue association: {known}    RFI flag: {candidate['rfi_like']}\n"
          "Zero acceleration; wrap-contaminated tail excluded. Fold/refinement statistics are diagnostic, not calibrated detection significance.")
    ax.text(0,1,text,va='top',fontsize=10,linespacing=1.6)
    title='Periodic candidate'
    if matches:title+=' — '+matches[0]['name']+' catalogue match'
    fig.suptitle(title+'\n'+Path(metadata['filename']).stem,fontsize=14)
    partial=path.with_name(path.stem+'.partial.png');fig.savefig(partial,dpi=140);plt.close(fig);os.replace(partial,path)


def fold_candidates(candidates,trial_dir,output_dir,metadata,config):
    from .periodicity import valid_trial
    # Peaks the cross-beam veto found in many beams never take a fold slot.
    best=sorted((r for r in candidates if r['is_sifted_best'] and not r.get('multibeam_rfi')),
                key=lambda r:(r['rfi_like'],-r['statistic']))[:config['max_folds']]
    if not best:return []
    catalogue,status=catalogue_matches(metadata) if config['catalogue_match'] else ([], 'disabled')
    directory=Path(output_dir)/'periodicity_plots';directory.mkdir(exist_ok=True)
    output=[]
    for rank,candidate in enumerate(best):
        ds=int(round(candidate['effective_sampling_seconds']/metadata['tsamp']))
        values,_=valid_trial(Path(trial_dir)/candidate['trial_file'],metadata,candidate['dm'],ds)
        fold=refine_fold(values,candidate['effective_sampling_seconds'],candidate,config)
        matches=match_catalogue(candidate,fold['period'],catalogue)
        row=dict(candidate,catalogue_matches=matches,catalogue_status=status,
                 refined_period_seconds=fold['period'],refined_frequency_hz=fold['frequency'],
                 fold_chi2=fold['chi2'],fold_bins=fold['bins'],review_status='unreviewed')
        near=[r for r in candidates if abs(r['period_seconds']/candidate['period_seconds']-1)<config['sift_period_fraction']]
        curve=np.array([(r['dm'],r['statistic']) for r in near]).reshape(-1,2)
        subbands=limits=None
        # A frequency-phase panel for the top non-RFI candidate; all folds keep
        # their time-phase, profile and refinement data. No filterbank reread for
        # each DM trial or each lower-ranked candidate.
        if rank==0 and not row['rfi_like'] and Path(metadata['filename']).is_file():
            subbands,limits=fold_subbands(metadata['filename'],row['dm'],fold['frequency'],fold['bins'],
                                         row['observation_seconds'],metadata.get('bad_channels',[]))
        stem=f"periodic_{rank+1:03d}_DM{row['dm']:.3f}_P{fold['period']:.9f}"
        plot=directory/(stem+'.png'); data=directory/(stem+'.npz')
        with data.with_suffix('.npz.partial').open('wb') as stream:
            np.savez_compressed(stream,profile=fold['profile'],errors=fold['errors'],
                subintegration_sums=fold['sums'],subintegration_counts=fold['counts'],
                frequencies=fold['frequencies'],fold_chi2=fold['scores'],dm_curve=curve,
                subbands=subbands if subbands is not None else np.empty((0,0)))
        os.replace(data.with_suffix('.npz.partial'),data)
        plot_fold(plot,fold,row,metadata,curve,subbands,limits)
        row.update(plot=str(plot.relative_to(output_dir)),fold_data=str(data.relative_to(output_dir)))
        output.append(row)
    return output
