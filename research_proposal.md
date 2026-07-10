Title (≤15 words)
EEG Microstate Degradation in Subjective Cognitive Decline and Mild Cognitive Impairment

Executive Summary (≤100 words / ≤3 sentences)
This study asks a simple question: do resting-state EEG microstates change in an orderly way from healthy aging to subjective cognitive decline (SCD) and then to mild cognitive impairment (MCI)? Using an openly documented pipeline, we will extract standard microstate temporal features and test both group differences and basic classification with interpretable models. If microstate “degradation” can be reproduced across participants, it would support EEG microstates as practical, low-cost markers for early risk along the Alzheimer’s disease continuum.

Introduction (200–300 words)
Alzheimer’s disease (AD) is rarely a sudden event; it is better understood as a long trajectory. Long before dementia is diagnosed, some people begin to notice subtle memory or attention problems even when routine testing still looks normal. This stage is often described as subjective cognitive decline (SCD). A later, more measurable stage is mild cognitive impairment (MCI), where cognitive performance is objectively reduced but everyday independence is largely preserved. From a public health perspective, these stages matter because they represent a window in which identification and monitoring may be most useful.

EEG offers an attractive tool for this purpose. It is comparatively inexpensive, non-invasive, and directly reflects neuronal activity at a millisecond time scale. Importantly, professional recommendations consider EEG and ERP measures relevant to research and assessment in dementia and MCI, which motivates careful, reproducible EEG biomarker studies (Babiloni et al., 2020). Within EEG analysis, microstates provide a compact way to describe fast-changing brain dynamics: brief periods in which the scalp voltage topography remains stable, followed by rapid switching to another topography (Michel & Koenig, 2018). A standard workflow clusters these topographies into a small set of recurring states and summarizes how long each state lasts, how often it appears, and how states transition (Khanna et al., 2015).

Prior work links microstate abnormalities to AD-spectrum conditions (Nishida et al., 2013). More recently, microstate pattern degradation has been reported already in SCD and MCI, suggesting potential utility as early biomarkers (Degradation of EEG microstates patterns…, 2023). This project aims to test whether those findings can be reproduced with a transparent analysis pipeline and whether microstate features carry enough signal to support simple, interpretable classification.

Proposed Methodology (200–300 words)
We will perform a secondary analysis of resting-state EEG data with diagnostic labels for healthy controls and MCI, and include SCD when available. Public datasets will be prioritized, and we will select a dataset only if the recording condition is clearly described (eyes-closed preferred), channel layout is adequate (≥19 channels recommended), and participant metadata are usable (age/sex; cognitive scores such as MMSE/MoCA if provided). If SCD labels are not available, the main analysis will focus on HC vs MCI, with SCD treated as an extension rather than a requirement.

Preprocessing will follow a fixed, documented pipeline (e.g., in MNE-Python): 1–40 Hz band-pass filtering, inspection and handling of bad channels, ICA-based removal of ocular and muscle artifacts, and average re-referencing. To avoid “more data = more noise,” we will standardize the analyzed duration (e.g., a clean 3–5 minute segment per participant).

For microstate analysis, we will compute global field power (GFP) and extract scalp maps at GFP peaks. We will then apply k-means clustering to derive microstate templates, using a four-state solution as the primary setting for comparability with common practice, and optionally checking robustness with alternative k values (Khanna et al., 2015; Michel & Koenig, 2018). After back-fitting templates to continuous EEG, we will extract microstate features: mean duration, occurrence rate, time coverage, and transition probabilities.

Statistical comparisons will use ANOVA or nonparametric tests depending on distributional checks, with FDR correction for multiple testing. For predictive evaluation, we will train interpretable models (logistic regression, linear SVM, and/or random forest) with stratified k-fold cross-validation and report balanced accuracy and ROC-AUC, alongside simple feature-importance summaries.
This project investigates whether resting-state EEG microstate features can differentiate healthy controls, subjective cognitive decline (SCD), and mild cognitive impairment (MCI). Using a reproducible preprocessing and microstate-clustering pipeline, we will extract standard temporal microstate parameters and test group differences and classification performance with interpretable machine learning. The goal is to evaluate microstates as low-cost, non-invasive early biomarkers along the Alzheimer’s disease continuum.
Timeline (schedule form)
Period	Tasks	Deliverables
Weeks 1–2	Finalize dataset choice, set up software environment, define inclusion/exclusion	Dataset list; analysis plan
Weeks 3–6	Preprocessing pipeline + quality control	Cleaned EEG segments; QC report
Weeks 7–10	Microstate clustering and back-fitting	Microstate templates; subject microstate sequences
Weeks 11–14	Feature extraction + statistical analysis	Feature table; group comparison figures/tables
Weeks 15–18	Machine learning modeling + validation	Cross-validated metrics; model interpretation
Weeks 19–20	Writing and revision	Final proposal/report; reproducible code package
References
Babiloni, C., Barry, R. J., Başar, E., Blinowska, K. J., Cichocki, A., Drinkenburg, W. H. I. M., 
Klimesch, W., Knight, R. T., Lopes da Silva, F. H., Nunez, P., Oostenveld, R., Jeong, J., 
Pascual-Marqui, R. D., Valdes-Sosa, P., Hallett, M., & Rossini, P. M. (2020). International 
Federation of Clinical Neurophysiology (IFCN)–EEG research workgroup: Recommendations on EEG and ERP in dementia and mild cognitive impairment.Clinical Neurophysiology, 131(10), 2458–2475. https://doi.org/10.1016/j.clinph.2019.06.234

Degradation of EEG microstates patterns in subjective cognitive decline and mild cognitive impairment: Early biomarkers along the Alzheimer’s Disease continuum? (2023).NeuroImage: Clinical, 38, 103407.

Khanna, A., Pascual-Leone, A., Michel, C. M., & Farzan, F. (2015). Microstates in resting-state EEG: Current status and future directions.Neuroscience & Biobehavioral Reviews, 49, 105–113.
	
Michel, C. M., & Koenig, T. (2018). EEG microstates as a tool for studying the temporal dynamics of whole-brain neuronal networks: A review.NeuroImage, 180, 577–593.

Nishida, K., Morishima, Y., Yoshimura, M., Isotani, T., Irisawa, S., Jann, K., Dierks, T., Kinoshita, T., & Koenig, T. (2013). EEG microstates associated with Alzheimer’s disease in elderly subjects.Clinical Neurophysiology, 124(6), 1075–1084.
