from pathlib import Path
import pandas as pd
R=Path(__file__).resolve().parents[1]; F=R/"data/raw/online_retail_II.xlsx"; O=R/"data/processed/clean_transactions.csv"
def main():
    if not F.exists(): raise FileNotFoundError(f"Put online_retail_II.xlsx here: {F}")
    d=pd.concat([pd.read_excel(F,sheet_name=s) for s in ["Year 2009-2010","Year 2010-2011"]],ignore_index=True)
    d.columns=[str(c).strip() for c in d.columns]
    need=["Invoice","StockCode","Quantity","InvoiceDate","Price","Customer ID","Country"]
    miss=[c for c in need if c not in d]
    if miss: raise ValueError(f"Missing columns: {miss}")
    d["Invoice"]=d["Invoice"].astype("string").str.strip()
    d["InvoiceDate"]=pd.to_datetime(d["InvoiceDate"],errors="coerce")
    for c in ["Customer ID","Quantity","Price"]: d[c]=pd.to_numeric(d[c],errors="coerce")
    d["Revenue"]=d["Quantity"]*d["Price"]
    n=len(d)
    d=d.dropna(subset=need)
    d=d[~d["Invoice"].str.upper().str.startswith("C",na=False)]
    d=d[(d["Quantity"]>0)&(d["Price"]>0)&(d["Revenue"]>0)].copy()
    d["Customer ID"]=d["Customer ID"].astype("int64"); d["StockCode"]=d["StockCode"].astype(str)
    d["Country"]=d["Country"].astype(str).str.strip()
    d=d.sort_values("InvoiceDate")
    O.parent.mkdir(parents=True,exist_ok=True); d.to_csv(O,index=False)
    print(f"Rows before cleaning: {n:,}\nRows after cleaning: {len(d):,}\nCustomers: {d['Customer ID'].nunique():,}\nSaved: {O}")
if __name__=="__main__": main()
