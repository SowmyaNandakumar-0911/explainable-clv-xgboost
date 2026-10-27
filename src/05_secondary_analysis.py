from pathlib import Path
import json, joblib
import numpy as np, pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score,roc_auc_score
R=Path(__file__).resolve().parents[1]; P=R/"data/processed"; M=R/"models"; Q=R/"reports"
F=["recency_days","tenure_days","frequency","monetary_total","monetary_avg","monetary_std","avg_qty","product_diversity","active_months","is_uk"]
def metrics(y,p,prob=None):
    z={"accuracy":float(accuracy_score(y,p)),"F1":float(f1_score(y,p,zero_division=0)),"precision":float(precision_score(y,p,zero_division=0)),"recall":float(recall_score(y,p,zero_division=0))}
    if prob is not None and len(np.unique(y))>1: z["AUC"]=float(roc_auc_score(y,prob))
    return z
def main():
    a=pd.read_csv(P/"train_data.csv"); b=pd.read_csv(P/"test_data.csv")
    xt=a[F].astype(float); xv=b[F].astype(float); yt=(a.future_spend_raw>0).astype(int); yv=(b.future_spend_raw>0).astype(int)
    clf=XGBClassifier(objective="binary:logistic",eval_metric="logloss",n_estimators=400,learning_rate=.03,max_depth=3,min_child_weight=3,subsample=.85,colsample_bytree=.85,reg_lambda=5,random_state=42,n_jobs=-1).fit(xt,yt)
    prob=clf.predict_proba(xv)[:,1]; pred=(prob>=.5).astype(int)
    model=joblib.load(M/"xgboost_clv.joblib"); cap=float(a.future_spend_capped.max()); spend=np.clip(model.predict(xv),0,cap); actual=b.future_spend_raw.to_numpy()
    out={"note":"Secondary analyses only; do not substitute for primary regression metrics.","will_spend_again":metrics(yv,pred,prob)}
    for pct in [20,10]:
        threshold=float(a.future_spend_raw.quantile(1-pct/100)); out[f"top_{pct}_percent_value_identification"]={"training_threshold":threshold,**metrics((actual>=threshold).astype(int),(spend>=threshold).astype(int))}
    joblib.dump(clf,M/"xgboost_will_spend_again_classifier.joblib"); Q.mkdir(exist_ok=True); (Q/"secondary_metrics.json").write_text(json.dumps(out,indent=2),encoding="utf-8")
    print(json.dumps(out,indent=2))
if __name__=="__main__": main()
