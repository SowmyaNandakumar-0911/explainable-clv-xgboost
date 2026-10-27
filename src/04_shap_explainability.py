from pathlib import Path
import joblib, pandas as pd, shap, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1]; M=R/"models/xgboost_clv.joblib"; I=R/"data/processed/test_data.csv"; O=R/"reports/figures/shap_summary.png"
F=["recency_days","tenure_days","frequency","monetary_total","monetary_avg","monetary_std","avg_qty","product_diversity","active_months","is_uk"]
def main():
    if not M.exists() or not I.exists(): raise FileNotFoundError("Run src/03_train_model.py first.")
    d=pd.read_csv(I); x=d[F].astype(float).sample(n=min(1000,len(d)),random_state=42)
    values=shap.TreeExplainer(joblib.load(M)).shap_values(x)
    O.parent.mkdir(parents=True,exist_ok=True); shap.summary_plot(values,x,show=False); plt.tight_layout(); plt.savefig(O,dpi=200,bbox_inches="tight"); plt.close()
    print(f"Saved: {O}")
if __name__=="__main__": main()
