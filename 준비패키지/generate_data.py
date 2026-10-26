"""Reproducible software test data, not a calibrated metallurgical simulator."""
from pathlib import Path
import json, hashlib
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
CFG=json.loads((ROOT/'config.json').read_text(encoding='utf-8'))

def generate():
    rng=np.random.default_rng(CFG['seed'])
    proc=[]; inspections=[]; labels=[]
    for run in range(CFG['n_lots']//CFG['lots_per_run']):
        # Correlated process variation and occasional excursions are assumed, not measured.
        regime=int(rng.choice(3,p=[.76,.14,.10]))
        heat=rng.normal(840 if regime!=1 else 808,12)
        temper=rng.normal(180 if regime!=2 else 240,18)
        oil=rng.normal(60,11)
        load=np.clip(rng.normal(100,24),35,180)
        shared_noise=rng.normal(0,.55)
        for lot_in_run in range(CFG['lots_per_run']):
            k=run*CFG['lots_per_run']+lot_in_run
            lot=f'SYN-L{k+1:04d}'
            values={
                'austenitize_mean_c':heat+rng.normal(0,1.5),
                'austenitize_dev_c':np.clip(abs(heat-840)+abs(rng.normal(4,2)),0,95),
                'hold_min':np.clip(rng.normal(35+(load-100)*.05,7),8,75),
                'quench_oil_c':np.clip(oil+rng.normal(0,1),15,120),
                'transfer_sec':np.clip(rng.normal(8+(load-100)*.025,3),1,30),
                'temper_mean_c':np.clip(temper+rng.normal(0,2),110,330),
                'temper_hold_min':np.clip(rng.normal(90,17),30,160),
                'load_kg':load,
            }
            values={key:round(float(value),3) for key,value in values.items()}
            v=values
            # Demonstration assumptions: quadratic heating response, tempering softening,
            # cooling/transfer interaction, shared run effect, unobserved variation.
            latent=(63.5-.0035*(v['austenitize_mean_c']-840)**2
                -.026*max(v['temper_mean_c']-180,0)+.035*max(180-v['temper_mean_c'],0)
                -.02*abs(v['hold_min']-35)-.055*max(v['quench_oil_c']-65,0)
                -.11*max(v['transfer_sec']-10,0)-.006*max(v['load_kg']-120,0)
                -.008*max(v['temper_hold_min']-90,0)
                -.002*max(v['transfer_sec']-10,0)*max(v['quench_oil_c']-65,0)
                +shared_noise+rng.normal(0,.35))
            hs=np.round(latent+rng.normal(0,.25,CFG['samples_per_lot']),3)
            target=int(np.any((hs<CFG['hardness_lower_hrc'])|(hs>CFG['hardness_upper_hrc'])))
            proc.append({'lot_id':lot,'heat_run_id':f'SYN-R{run+1:04d}',
                'process_end_at':(datetime(2026,1,1)+timedelta(hours=6*run)).isoformat(),
                'material_grade':CFG['material_grade'],'diameter_mm':CFG['diameter_mm'],
                'recipe_id':'DEMO-HT-01',**values,'data_source':'synthetic_demo'})
            for n,h in enumerate(hs):
                inspections.append({'lot_id':lot,'sample_id':f'{lot}-S{n+1}',
                    'hardness_hrc':float(h),'lower_hrc':CFG['hardness_lower_hrc'],
                    'upper_hrc':CFG['hardness_upper_hrc'],'data_source':'synthetic_demo'})
            labels.append({'lot_id':lot,'sample_count':len(hs),'hardness_min_hrc':float(hs.min()),
                'hardness_max_hrc':float(hs.max()),'lot_nonconforming':target,
                'judgment_rule':'DEMO_ANY_OF_5_OUTSIDE_60_66','data_source':'synthetic_demo'})
    data=ROOT/'data'; data.mkdir(exist_ok=True)
    frames={'process':pd.DataFrame(proc),'inspection':pd.DataFrame(inspections),'labels':pd.DataFrame(labels)}
    for name,df in frames.items():df.to_csv(data/f'{name}.csv',index=False,encoding='utf-8-sig')
    korean={'lot_id':'로트번호','heat_run_id':'열처리작업','process_end_at':'공정완료시각',
        'material_grade':'재질','diameter_mm':'직경_mm','austenitize_mean_c':'가열평균_C',
        'austenitize_dev_c':'가열최대편차_C','hold_min':'가열유지_분','quench_oil_c':'냉각유온_C',
        'transfer_sec':'이송지연_초','temper_mean_c':'뜨임평균_C','temper_hold_min':'뜨임유지_분','load_kg':'장입량_kg'}
    demo=frames['process'].iloc[800:820].copy()
    demo.to_csv(data/'demo_upload.csv',index=False,encoding='utf-8-sig')
    demo.rename(columns=korean).to_csv(data/'demo_upload_korean.csv',index=False,encoding='utf-8-sig')
    (data/'mapping_korean.json').write_text(json.dumps({v:k for k,v in korean.items()},ensure_ascii=False,indent=2),encoding='utf-8')
    bad=demo.iloc[:6].copy()
    bad.loc[bad.index[1],'material_grade']='OTHER'
    bad.loc[bad.index[2],'diameter_mm']=12
    bad.loc[bad.index[3],'austenitize_mean_c']=None
    bad.loc[bad.index[4],'lot_id']=bad.iloc[0]['lot_id']
    bad.loc[bad.index[5],'temper_mean_c']=9999
    bad.to_csv(data/'demo_errors.csv',index=False,encoding='utf-8-sig')
    workbook={name:{'columns':list(df.columns),'rows':df.astype(object).where(pd.notna(df),None).values.tolist()} for name,df in frames.items()}
    (data/'workbook_data.json').write_text(json.dumps(workbook,ensure_ascii=False),encoding='utf-8')
    summary={'version':CFG['version'],'seed':CFG['seed'],'lots':len(proc),'heat_runs':500,
        'inspection_rows':len(inspections),'nonconforming_lots':int(frames['labels'].lot_nonconforming.sum()),
        'data_status':CFG['data_status'],'recipe_values_are_demo_assumptions':True,
        'sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in data.glob('*.csv')}}
    (data/'manifest.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__': generate()
