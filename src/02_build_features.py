from pathlib import Path
import pandas as pd
R=Path(__file__).resolve().parents[1]; I=R/"data/processed/clean_transactions.csv"; O=R/"data/processed/modeling_table.csv"
CUT=pd.Timestamp("2011-06-09"); END=pd.Timestamp("2011-12-09 23:59:59")
F=["recency_days","tenure_days","frequency","monetary_total","monetary_avg","monetary_std","avg_qty","product_diversity","active_months","is_uk"]
def main():
    if not I.exists(): raise FileNotFoundError("Run src/01_clean_data.py first.")
    d=pd.read_csv(I,parse_dates=["InvoiceDate"]); h=d[d.InvoiceDate<CUT].copy(); f=d[(d.InvoiceDate>=CUT)&(d.InvoiceDate<=END)].copy()
    inv=h.groupby(["Customer ID","Invoice"],as_index=False).agg(order_spend=("Revenue","sum"),order_qty=("Quantity","sum"))
    os=inv.groupby("Customer ID").agg(monetary_avg=("order_spend","mean"),monetary_std=("order_spend","std"),avg_qty=("order_qty","mean"))
    os["monetary_std"]=os["monetary_std"].fillna(0)
    x=h.groupby("Customer ID").agg(first=("InvoiceDate","min"),last=("InvoiceDate","max"),frequency=("Invoice","nunique"),monetary_total=("Revenue","sum"),product_diversity=("StockCode","nunique"),active_months=("InvoiceDate",lambda s:s.dt.to_period("M").nunique()),country=("Country",lambda s:s.mode().iloc[0] if not s.mode().empty else s.iloc[0]))
    x["recency_days"]=(CUT-x["last"]).dt.days; x["tenure_days"]=(x["last"]-x["first"]).dt.days
    x["is_uk"]=(x["country"].str.lower().str.strip()=="united kingdom").astype(int); x=x.join(os)[F]
    y=f.groupby("Customer ID").Revenue.sum().rename("future_spend")
    out=x.join(y).fillna({"future_spend":0}).reset_index().dropna(subset=F+["future_spend"])
    out["Customer ID"]=out["Customer ID"].astype("int64"); O.parent.mkdir(parents=True,exist_ok=True); out.to_csv(O,index=False)
    print(f"Customers with history: {len(out):,}\nPositive future spend: {(out.future_spend>0).sum():,}\nSaved: {O}")
if __name__=="__main__": main()
