"""Small-sample, chronological pilot. Results are not deployment validation."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,precision_score,recall_score,f1_score,fbeta_score,roc_auc_score,average_precision_score
from sklearn.inspection import permutation_importance
from xgboost import XGBClassifier

ROOT=Path(__file__).resolve().parent/'kamp'

def metrics(y,p,threshold):
    pred=np.asarray(p)>=threshold
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return {'count':len(y),'positive_count':int(sum(y)),'tp':int(tp),'fn':int(fn),'fp':int(fp),'tn':int(tn),
            'precision':float(precision_score(y,pred,zero_division=0)),
            'recall':float(recall_score(y,pred,zero_division=0)),
            'f1':float(f1_score(y,pred,zero_division=0)),'f2':float(fbeta_score(y,pred,beta=2,zero_division=0)),
            'roc_auc':float(roc_auc_score(y,p)) if len(set(y))>1 else None,
            'average_precision':float(average_precision_score(y,p)) if sum(y) else None}

def main():
    df=pd.read_csv(ROOT/'joined_batches.csv',encoding='utf-8-sig',parse_dates=['process_start_at','process_end_at'])
    audit=json.loads((ROOT/'audit.json').read_text(encoding='utf-8'))
    features=audit['features']
    # Calendar boundaries chosen before examining model scores. Purge crossing batches.
    b1=pd.Timestamp('2022-05-01');b2=pd.Timestamp('2022-06-01')
    df['split']='boundary_purged'
    df.loc[df.process_end_at<b1,'split']='train'
    df.loc[(df.process_start_at>=b1)&(df.process_end_at<b2),'split']='validation'
    df.loc[df.process_start_at>=b2,'split']='test'
    train=df[df.split=='train'].copy(); val=df[df.split=='validation'].copy(); test=df[df.split=='test'].copy()
    assert train.process_end_at.max()<val.process_start_at.min()
    assert val.process_end_at.max()<test.process_start_at.min()
    assert len(set(train.batch_id)&set(val.batch_id))==len(set(val.batch_id)&set(test.batch_id))==0
    rate_threshold=float(train.defect_rate_percent.quantile(.75))
    df['risk_label']=(df.defect_rate_percent>=rate_threshold).astype(int)
    for d in [train,val,test]:d['risk_label']=(d.defect_rate_percent>=rate_threshold).astype(int)
    y=train.risk_label
    assert y.nunique()==2 and val.risk_label.nunique()==2
    models={
      'LogisticRegression':LogisticRegression(C=.1,class_weight='balanced',max_iter=3000,random_state=20261026),
      'RandomForest':RandomForestClassifier(n_estimators=200,max_depth=3,min_samples_leaf=6,class_weight='balanced',n_jobs=2,random_state=20261026),
      'XGBoost':XGBClassifier(n_estimators=80,max_depth=2,learning_rate=.05,min_child_weight=3,reg_lambda=5,
                             scale_pos_weight=float((y==0).sum()/(y==1).sum()),n_jobs=2,random_state=20261026,eval_metric='logloss')}
    fitted={};comparison=[]
    for name,m in models.items():
        pipe=Pipeline([('imputer',SimpleImputer(strategy='median')),('scale',StandardScaler() if name=='LogisticRegression' else 'passthrough'),('model',m)])
        pipe.fit(train[features],y);p=pipe.predict_proba(val[features])[:,1]
        choices=[(float(t),metrics(val.risk_label,p,float(t))) for t in np.arange(.1,.91,.05)]
        t,score=max(choices,key=lambda x:(x[1]['f2'],x[1]['precision'],x[0]))
        comparison.append({'model':name,'alert_threshold':round(t,6),**score});fitted[name]=pipe
    best=max(comparison,key=lambda x:(x['f2'],x['precision']))
    name=best['model'];pipe=fitted[name];t=best['alert_threshold']
    # Holdout is evaluated once, only after selecting model and threshold using validation.
    test_scores=pipe.predict_proba(test[features])[:,1]
    result=metrics(test.risk_label,test_scores,t)
    importance=permutation_importance(pipe,val[features],val.risk_label,scoring='average_precision',n_repeats=10,random_state=20261026,n_jobs=1)
    order=np.argsort(importance.importances_mean)[::-1][:6]
    reference={'median':train[features].median().to_dict(),'q05':train[features].quantile(.05).to_dict(),'q95':train[features].quantile(.95).to_dict()}
    model_bundle={'pipeline':pipe,'features':features,'alert_threshold':t,'defect_rate_threshold_percent':rate_threshold,
                  'version':'kamp60-pilot-20261026','data_source':'KAMP_heat_quality_60','reference':reference}
    joblib.dump(model_bundle,ROOT/'pilot_model.joblib')
    splits={k:{'batches':len(g),'risk_labels':int(g.risk_label.sum()),'start':str(g.process_start_at.min()),'end':str(g.process_end_at.max())} for k,g in df.groupby('split')}
    report={'scope':'Automotive-parts salt-bath heat treatment; batch-level relative high-defect risk',
            'label_definition':'defect_rate_percent >= training-set Q3; a relative pilot label, not product acceptance',
            'defect_rate_threshold_percent':rate_threshold,'defect_rate_threshold_ppm':rate_threshold*10000,
            'split':splits,'validation_comparison':comparison,'selected_model':name,'alert_threshold':t,
            'test':result,'test_all_normal':metrics(test.risk_label,np.zeros(len(test)),.5),
            'test_all_risk':metrics(test.risk_label,np.ones(len(test)),.5),
            'validation_all_risk':metrics(val.risk_label,np.ones(len(val)),.5),
            'feature_importance_validation':[{'feature':features[i],'average_precision_drop':float(importance.importances_mean[i])} for i in order],
            'limitations':['Only 136 labeled batches; millions of sensor records are not independent labels.',
                           'Features summarize the complete batch: prediction is after completion, before using inspection results.',
                           'No measured hardness; no validation of the synthetic steel-ball model.',
                           'Risk threshold is relative to training Q3, not an industry reject specification.',
                           'No causal inference or calibrated defect probability. CPU-only offline pilot.',
                           'Inspection timestamp, product mix and material are unavailable; verify operational timing with mentor.'],
            'source_url':audit['source_url'],'raw_audit':'audit.json','feature_count':len(features)}
    (ROOT/'benchmark.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    pd.DataFrame(comparison).to_csv(ROOT/'validation_comparison.csv',index=False,encoding='utf-8-sig')
    df[['batch_id','process_start_at','process_end_at','split','defect_rate_percent','risk_label']].to_csv(ROOT/'split_manifest.csv',index=False,encoding='utf-8-sig')
    prediction=test[['batch_id','defect_rate_percent','risk_label']].copy()
    prediction['risk_score']=test_scores;prediction['predicted_risk']=(test_scores>=t).astype(int)
    prediction.to_csv(ROOT/'holdout_predictions.csv',index=False,encoding='utf-8-sig')
    test[['batch_id','process_start_at','process_end_at','sensor_rows','data_source','missing_fraction',*features]].head(20).to_csv(ROOT/'demo_upload.csv',index=False,encoding='utf-8-sig')
    print(json.dumps({k:report[k] for k in ['split','defect_rate_threshold_percent','selected_model','alert_threshold','test','test_all_risk']},ensure_ascii=False))

if __name__=='__main__':main()
