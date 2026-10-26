from pathlib import Path
import json, sys, hashlib, importlib.metadata
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,fbeta_score,confusion_matrix,roc_auc_score
from sklearn.inspection import permutation_importance
from xgboost import XGBClassifier
import joblib

ROOT=Path(__file__).resolve().parent
C=json.loads((ROOT/'config.json').read_text(encoding='utf-8'))

def metrics(y,score,threshold):
    predicted=(score>=threshold).astype(int)
    tn,fp,fn,tp=confusion_matrix(y,predicted,labels=[0,1]).ravel()
    return {'n':len(y),'nonconforming_n':int(sum(y)), 'threshold':float(threshold),
        'accuracy':float(accuracy_score(y,predicted)), 'precision':float(precision_score(y,predicted,zero_division=0)),
        'recall':float(recall_score(y,predicted,zero_division=0)), 'f1':float(f1_score(y,predicted,zero_division=0)),
        'f2':float(fbeta_score(y,predicted,beta=2,zero_division=0)),
        'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp)}

def train():
    df=pd.read_csv(ROOT/'data/process.csv').merge(pd.read_csv(ROOT/'data/labels.csv')[['lot_id','lot_nonconforming']],on='lot_id',validate='one_to_one')
    df=df.sort_values(['process_end_at','heat_run_id','lot_id']).reset_index(drop=True)
    a,b,c=C['split_lots']; splits={'train':df.iloc[:a],'validation':df.iloc[a:a+b],'test':df.iloc[a+b:]}
    groups={k:set(d.heat_run_id) for k,d in splits.items()}
    assert not(groups['train']&groups['validation'] or groups['train']&groups['test'] or groups['validation']&groups['test'])
    for d in splits.values():assert d.lot_nonconforming.nunique()==2
    X,y=splits['train'][C['features']],splits['train'].lot_nonconforming
    VX,VY=splits['validation'][C['features']],splits['validation'].lot_nonconforming
    candidates={
        'logistic_regression':make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),LogisticRegression(max_iter=1500,class_weight='balanced',random_state=C['seed'])),
        'random_forest':make_pipeline(SimpleImputer(strategy='median'),RandomForestClassifier(n_estimators=180,max_depth=7,min_samples_leaf=4,class_weight='balanced',n_jobs=2,random_state=C['seed'])),
        'xgboost':make_pipeline(SimpleImputer(strategy='median'),XGBClassifier(n_estimators=180,max_depth=3,learning_rate=.05,subsample=.9,colsample_bytree=.9,n_jobs=2,eval_metric='logloss',random_state=C['seed']))}
    results=[]
    for name,model in candidates.items():
        model.fit(X,y); scores=model.predict_proba(VX)[:,1]
        options=[metrics(VY,scores,t) for t in np.arange(.10,.901,.05)]
        best=max(options,key=lambda m:(m['f2'],m['precision'],m['threshold']))
        results.append({'model':name,**best})
    selected=max(results,key=lambda m:(m['f2'],m['precision']))
    model=candidates[selected['model']]; threshold=selected['threshold']
    # Only the selected candidate is evaluated on the locked chronological test set.
    TX,TY=splits['test'][C['features']],splits['test'].lot_nonconforming
    test=metrics(TY,model.predict_proba(TX)[:,1],threshold)
    baseline=metrics(TY,np.full(len(TY),float(y.mean()>=.5)),.5)
    imp=permutation_importance(model,VX,VY,scoring='roc_auc',n_repeats=5,random_state=C['seed'],n_jobs=1)
    importance=sorted([{'feature':f,'validation_auc_drop':float(v)} for f,v in zip(C['features'],imp.importances_mean)],key=lambda a:a['validation_auc_drop'],reverse=True)
    modeldir=ROOT/'models'; modeldir.mkdir(exist_ok=True)
    bundle={'pipeline':model,'threshold':threshold,'features':C['features'],'name':selected['model'],'version':C['version'],'data_status':C['data_status']}
    joblib.dump(bundle,modeldir/'selected_model.joblib')
    splitrows=df[['lot_id','heat_run_id','process_end_at']].copy()
    splitrows['split']=['train']*a+['validation']*b+['test']*c
    splitrows.to_csv(ROOT/'data/split_manifest.csv',index=False,encoding='utf-8-sig')
    versions={p:importlib.metadata.version(p) for p in ['numpy','pandas','scikit-learn','xgboost','fastapi','uvicorn','python-multipart','httpx','joblib','openpyxl']}
    report={'scope':'SYNTHETIC_FUNCTIONAL_BENCHMARK_ONLY','selection_rule':'Validation F2 over fixed threshold grid 0.10..0.90 step 0.05; precision tie break; test used only after selection',
        'splits':{k:{'lots':len(d),'heat_runs':len(groups[k]),'nonconforming':int(d.lot_nonconforming.sum())} for k,d in splits.items()},
        'validation_comparison':results,'selected_model':selected['model'],'selected_threshold':threshold,
        'test_selected_model':test,'test_majority_baseline':baseline,'global_importance':importance,'versions':versions,
        'model_sha256':hashlib.sha256((modeldir/'selected_model.joblib').read_bytes()).hexdigest()}
    out=ROOT/'results'; out.mkdir(exist_ok=True)
    (out/'benchmark.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    pd.DataFrame(results).to_csv(out/'validation_comparison.csv',index=False,encoding='utf-8-sig')
    (ROOT/'requirements.txt').write_text('\n'.join(f'{k}=={v}' for k,v in versions.items())+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':train()
