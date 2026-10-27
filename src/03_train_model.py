from pathlib import Path
import json, joblib
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from xgboost import XGBRegressor
R=Path(__file__).resolve().parents[1]; I=R/"data/processed/modeling_table.csv"; P=R/"data/processed"; M=R/"models"; Q=R/"reports"
F=["recency_days","tenure_days","frequency","monetary_total","monetary_avg","monetary_std","avg_qty","product_diversity","active_months","is_uk"]
def score(y,p): return {"R2":float(r2_score(y,p)),"MAE":float(mean_absolute_error(y,p)),"RMSE":float(np.sqrt(mean_squared_error(y,p)))}
def main():
    if not I.exists(): raise FileNotFoundError("Run src/02_build_features.py first.")
    d=pd.read_csv(I); X=d[F].astype(float); y=d.future_spend.astype(float)
    Xtr,Xte,ytr0,yte0=train_test_split(X,y,test_size=.20,random_state=42)
    cap=float(ytr0.quantile(.99)); ytr=ytr0.clip(upper=cap); yte=yte0.clip(upper=cap)
    lr=LinearRegression().fit(Xtr,ytr); lp=np.clip(lr.predict(Xte),0,cap)
    xgb=XGBRegressor(objective="reg:squarederror",n_estimators=500,learning_rate=.03,max_depth=4,min_child_weight=3,subsample=.85,colsample_bytree=.85,reg_lambda=5,random_state=42,n_jobs=-1).fit(Xtr,ytr)
    xp=np.clip(xgb.predict(Xte),0,cap)
    result={"target":"future_spend capped at training-set 99th percentile","cap_quantile":.99,"training_cap":cap,"random_state":42,"train_rows":len(Xtr),"test_rows":len(Xte),"features":F,"linear_regression":score(yte,lp),"xgboost":score(yte,xp)}
    M.mkdir(exist_ok=True); Q.mkdir(exist_ok=True); P.mkdir(exist_ok=True)
    joblib.dump(xgb,M/"xgboost_clv.joblib"); joblib.dump(lr,M/"linear_regression_baseline.joblib")
    a=Xtr.copy(); a["Customer ID"]=d.loc[Xtr.index,"Customer ID"].values; a["future_spend_raw"]=ytr0.values; a["future_spend_capped"]=ytr.values; a.to_csv(P/"train_data.csv",index=False)
    b=Xte.copy(); b["Customer ID"]=d.loc[Xte.index,"Customer ID"].values; b["future_spend_raw"]=yte0.values; b["future_spend_capped"]=yte.values; b["linear_regression_prediction"]=lp; b["xgboost_prediction"]=xp; b.to_csv(P/"test_data.csv",index=False)
    (Q/"primary_metrics.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    pd.DataFrame([{"Model":"Linear Regression baseline",**result["linear_regression"]},{"Model":"XGBoost proposed model",**result["xgboost"]}]).to_csv(Q/"baseline_comparison.csv",index=False)
    print(f"Training-only cap: {cap:.2f}\n"+pd.DataFrame([{"Model":"Linear Regression",**result["linear_regression"]},{"Model":"XGBoost",**result["xgboost"]}]).to_string(index=False))
if __name__=="__main__": main()
