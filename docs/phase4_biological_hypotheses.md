# Phase 4 biological hypotheses (frozen before interaction testing)

This panel was fixed before inspecting Phase 4 interaction significance. It is hypothesis-driven and is not a replacement for the genome-wide primary test. Every listed gene is present as a unique protein-coding entry in the released study-era annotation and passes the pre-specified count filter.

## Biological questions and frozen genes

| Theme | Pre-specified genes | Rationale and directional expectation |
|---|---|---|
| Antimicrobial defense | GBP1, GBP2, GBP3, GBP4, GBP5 | IFN-inducible guanylate-binding proteins participate in cell-autonomous defense and cytosolic recognition. Priming could amplify infection-induced expression, but saturation after strong IFN signaling could also yield a negative log-scale interaction. No direction is required for inclusion. |
| Interferon signaling | STAT1, IRF1, IDO1, CXCL9, CXCL10, CXCL11, IFIT1, IFIT2, IFIT3, ISG15, OAS1 | The study's IFN-gamma and infection scheme converges on interferon-regulatory transcriptional programs. Positive interaction would indicate greater-than-additive induction on the analyzed scale; negative interaction could reflect shared-pathway saturation or feedback rather than absolute suppression. |
| Inflammatory signaling | TNF, IL1B, NFKBIA, CXCL8, CCL2, PTGS2 | Salmonella activates TLR-linked inflammatory programs. Priming might alter their amplitude in either direction. NFKBIA is included as inducible feedback as well as an NF-kappaB response marker. |
| Immunoregulatory feedback | SOCS3, DUSP1, IL10, ATF3, TNFAIP3 | These genes can limit or reshape cytokine and stress signaling. An interaction may reflect protective restraint, tolerance-like feedback, or altered timing; its sign is not labeled beneficial or harmful. |

## Annotation verification

The exact identifiers are stored in `configs/phase4_candidate_panel.csv`, together with chromosome, biotype, and the local verification status. Gene symbols are used for display; Ensembl gene IDs are the analysis keys. Ambiguous or duplicated symbols must not be silently collapsed.

## Source basis

1. Alasoo et al. describe the experimental system and the IFN-gamma/STAT1-IRF1 and Salmonella/TLR4-NF-kappaB-IRF3/type-I-interferon axes used to motivate the signaling groups: [Nature Genetics 50, 424-431 (2018)](https://doi.org/10.1038/s41588-018-0046-7), PMID [29379200](https://pubmed.ncbi.nlm.nih.gov/29379200/).
2. Human GBP1 can recognize cytosolic Gram-negative bacterial LPS and promote a caspase-4 activation platform, providing a mechanistic basis for the GBP group: [Santos et al., Nature Communications (2020)](https://doi.org/10.1038/s41467-020-16889-z).
3. SOCS proteins are established negative regulators of cytokine/JAK-STAT signaling, motivating SOCS3 as feedback rather than assuming it is simply pro- or anti-inflammatory: [Yoshimura et al., Nature Reviews Immunology (2007)](https://doi.org/10.1038/nri2093).
4. DUSP1 regulates MAPK-dependent inflammatory responses and is included with ATF3, TNFAIP3, and IL10 as a broader feedback/stress-response group: [Abraham and Clark, Annual Review of Immunology (2006)](https://doi.org/10.1146/annurev.immunol.24.021605.090720).

These references motivate the panel; they do not predict which genes must be significant in this dataset. Transcript abundance alone cannot establish protein activity, bacterial killing, clinical protection, or causality.

