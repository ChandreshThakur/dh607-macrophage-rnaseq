#!/usr/bin/env python
"""Run and save the required synthetic and balanced real-data PyDESeq2 pilots."""

from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biological_analysis.phase4_interaction import _fit_pydeseq, load_data  # noqa: E402
from preprocessing.normalization import expressed_filter  # noqa: E402


def synthetic_pilot() -> dict[str, object]:
    rng = np.random.default_rng(607)
    donors = [f"d{i:02}" for i in range(12)]
    rows = []
    for donor in donors:
        for condition, ifng, salmonella in [("A",0,0),("B",1,0),("C",0,1),("D",1,1)]:
            rows.append({"sample_id":f"{donor}_{condition}","donor":donor,"condition":condition,
                         "condition_name":{"A":"naive","B":"IFNg","C":"SL1344","D":"IFNg_SL1344"}[condition],
                         "IFNg":{"A":"naive","B":"primed","C":"naive","D":"primed"}[condition],
                         "SL1344":{"A":"control","B":"control","C":"infected","D":"infected"}[condition],
                         "IFNg_binary":ifng,"Salmonella_binary":salmonella})
    metadata = pd.DataFrame(rows)
    genes = [f"g{i:03}" for i in range(200)]
    base = rng.normal(np.log(400),.6,200)
    beta_ifng = rng.normal(.3,.15,200); beta_salmonella = rng.normal(.6,.2,200)
    beta_interaction = np.zeros(200); beta_interaction[:20]=np.log(2)*1.5; beta_interaction[20:40]=-np.log(2)*1.5
    donor_effect = dict(zip(donors,rng.normal(0,.18,len(donors))))
    matrix=[]
    for _,row in metadata.iterrows():
        mu=np.exp(base+donor_effect[row.donor]+beta_ifng*row.IFNg_binary+beta_salmonella*row.Salmonella_binary+
                  beta_interaction*row.IFNg_binary*row.Salmonella_binary)
        matrix.append(rng.poisson(rng.gamma(shape=12.5,scale=mu*.08)))
    counts=pd.DataFrame(np.asarray(matrix).T,index=genes,columns=metadata.sample_id,dtype=int)
    started=time.time()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _,results,columns,contrast=_fit_pydeseq(counts,metadata,n_cpus=1)
    positive=results.iloc[:20].log2FoldChange; negative=results.iloc[20:40].log2FoldChange
    return {"seed":607,"n_donors":12,"n_genes":200,"known_positive_genes":20,"known_negative_genes":20,
            "tested_coefficient":columns[int(np.flatnonzero(contrast)[0])],"elapsed_seconds":time.time()-started,
            "positive_sign_recovery":float((positive>0).mean()),"negative_sign_recovery":float((negative<0).mean()),
            "positive_median_log2_effect":float(positive.median()),"negative_median_log2_effect":float(negative.median()),
            "known_effects_fdr_0_05":int((results.iloc[:40].padj<.05).sum()),"warnings":[str(w.message) for w in caught],
            "passed":bool((positive>0).all() and (negative<0).all() and columns[int(np.flatnonzero(contrast)[0])]=="IFNg_binary:Salmonella_binary")}


def real_pilot() -> dict[str, object]:
    metadata,counts,_,_=load_data(ROOT)
    donors=sorted(metadata.donor.unique())[:12]; metadata=metadata[metadata.donor.isin(donors)].copy()
    mask=expressed_filter(counts,10,84); ids=counts.loc[mask,metadata.sample_id].apply(np.log1p).var(axis=1).nlargest(250).index
    started=time.time()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        dds,results,columns,contrast=_fit_pydeseq(counts.loc[ids],metadata,n_cpus=1)
    return {"donor_selection":"first 12 sorted donor IDs","gene_selection":"250 highest log1p-count variance genes passing primary filter",
            "n_samples":len(metadata),"n_donors":len(donors),"n_genes":len(ids),"n_design_columns":len(columns),
            "design_rank":int(np.linalg.matrix_rank(dds.obsm["design_matrix"])),
            "tested_coefficient":columns[int(np.flatnonzero(contrast)[0])],"elapsed_seconds":time.time()-started,
            "finite_effects":int(np.isfinite(results.log2FoldChange).sum()),"finite_p_values":int(np.isfinite(results.pvalue).sum()),
            "lfc_converged":int(dds.var["_LFC_converged"].sum()),"map_dispersion_converged":int(dds.var["_MAP_converged"].sum()),
            "warnings":[str(w.message) for w in caught],"passed":bool(np.isfinite(results.log2FoldChange).all() and np.isfinite(results.pvalue).all() and dds.var["_LFC_converged"].all())}


def main() -> None:
    for variable in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
        os.environ.setdefault(variable,"1")
    output={"synthetic":synthetic_pilot(),"real_balanced_pilot":real_pilot()}
    path=ROOT / "results" / "tables" / "interaction_count_model_pilot_diagnostics.json"
    path.write_text(json.dumps(output,indent=2),encoding="utf-8")
    print(json.dumps(output,indent=2))


if __name__ == "__main__":
    main()
