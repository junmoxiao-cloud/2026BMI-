# Literature Review: EEG for Cognitive Decline Detection and Neurofeedback Therapy

**Research Direction: EEG Real-time Cognitive Feedback for MCI/AD Continuum**
**Author: Yucheng Qiao | Supervisor: Prof. Neal Bangerter, Imperial College London BME**
**Date: July 2026**

---

## Part 1: Research Background and Objectives

This literature review addresses four key research questions:
1. EEG Features: Which EEG biomarkers differentiate healthy aging, SCD, MCI, and AD?
2. EEG Microstate: Can microstate degradation serve as an early biomarker?
3. Neurofeedback Therapy: Can real-time EEG feedback intervention improve cognition in MCI patients?
4. BCI Pipeline: How to design the full closed-loop system from signal acquisition to feedback?

---

## Part 2: EEG Biomarkers Along the AD Continuum

### 2.1 Resting-state EEG Rhythms and Cognitive Decline

#### Key Reference A: Babiloni et al. (2019)
Title: What electrophysiology tells us about Alzheimer's disease
Journal: Neurobiology of Aging | Citations: 281
URL: https://pure.amsterdamumc.nl/en/publications/f461808c-342e-42fe-8acf-9b63b8b98fba

Key Findings:
- AD patients show characteristic alpha power decrease (8-13 Hz) and theta power increase (4-8 Hz)
- This 'EEG slowing' correlates with synaptic loss and cholinergic system damage
- MCI shows early-stage versions of these changes, supporting EEG as a predictive biomarker

---

#### Key Reference B: Meghdadi et al. (2021)
Title: Resting state EEG biomarkers of cognitive decline associated with AD and MCI
Journal: PLoS ONE | Citations: 236
URL: https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0244180&type=printable

Key Findings:
- Alpha relative power and theta/alpha ratio are the most robust discriminative features in large samples
- ML classifiers achieve AUC 0.85+ for AD/MCI vs. controls
- Relevance to this project: provides reference benchmark for feature extraction and classification

---

#### Key Reference C: Babiloni et al. (2021)
Title: Measures of resting state EEG rhythms for clinical trials in Alzheimer's disease
Journal: Alzheimer's and Dementia | Citations: 206
URL: https://onlinelibrary.wiley.com/doi/pdfdirect/10.1002/alz.12311

Key Findings:
- International expert panel consensus: individual alpha frequency (IAF) and alpha/theta ratio as primary EEG endpoints
- Standardized recording protocol: eyes-closed resting, 19+ channels, 5 minutes
- Methodological significance: authoritative reference for EEG acquisition and feature selection in this project

---

### 2.2 EEG Functional Connectivity

Prado et al. (2022)
Title: Dementia ConnEEGtome: Towards multicentric harmonization of EEG connectivity in neurodegeneration
Journal: Int J Psychophysiology | Citations: 41
URL: https://doi.org/10.1016/j.ijpsycho.2021.12.008
Notes: Reports DMN connectivity decline along AD continuum; multicenter harmonization is key for future replication

---

Adebisi et al. (2024)
Title: EEG-Based Brain Functional Network Analysis for Differential Identification of Dementia
Journal: IEEE TNSRE | Citations: 33
URL: https://ieeexplore.ieee.org/ielx7/7333/4359219/10462208.pdf
Notes: Graph theory metrics (clustering coefficient, path length) for AD vs. FTD; complements microstate analysis

---

### 2.3 Latest qEEG Review

Yuan and Zhao (2025)
Title: The role of quantitative EEG biomarkers in AD and MCI: applications and insights
Journal: Frontiers in Aging Neuroscience | Citations: 32
URL: https://www.frontiersin.org/journals/aging-neuroscience/articles/10.3389/fnagi.2025.1522552/pdf
Notes: Comprehensive 2025 review of spectral, complexity, connectivity, and microstate biomarkers -- recommended as introductory reading

---

## Part 3: EEG Microstates -- Core Analysis Method

### 3.1 Microstate Theory

Michel and Koenig (2018) [already cited in current research proposal]
Title: EEG microstates as a tool for studying temporal dynamics of whole-brain neuronal networks
Journal: NeuroImage

Microstate Parameters and AD Continuum Changes:

| Parameter | Meaning | Change in AD |
|-----------|---------|--------------|
| Mean Duration | Time each microstate is maintained | Decreases -- network instability |
| Occurrence Rate | Occurrences of a state per unit time | Decrease in states C/D |
| Coverage | Fraction of total time in a state | Reflects cognitive state |
| Transition Probability | Pattern of state switching | More random transitions |

---

### 3.2 Microstate Degradation in the AD Continuum

### [CORE PAPER -- most directly relevant to this project]

Lassi et al. (2023)
Title: Degradation of EEG microstates patterns in SCD and MCI: Early biomarkers along the AD continuum?
Journal: NeuroImage: Clinical | Citations: 51
URL: https://doi.org/10.1016/j.nicl.2023.103407

Core Findings:
- Along HC -> SCD -> MCI continuum, microstate C duration and coverage progressively decrease
- SCD stage already shows statistically significant microstate degradation, preceding neuropsychological test changes
- Provides complete MNE-Python analysis pipeline reference

Replication Strategy for This Project:
- Use public datasets (TUAB, OpenNeuro ds004504)
- Reproduce the 4-state k-means clustering scheme
- Extension: add SVM/random forest classifiers to evaluate diagnostic value

---

Luo et al. (2020)
Title: Biomarkers for Prediction of Schizophrenia: Insights From Resting-State EEG Microstates
Journal: IEEE Access | Citations: 58
URL: https://ieeexplore.ieee.org/ielx7/6287639/8948470/09257371.pdf
Notes: Demonstrates microstate biomarker potential in psychiatric disorders; methodology transfers to AD

---

### 3.3 Full EEG Auto-Analysis Pipeline

Ding et al. (2021)
Title: Fully automated discrimination of Alzheimer's disease using resting-state EEG signals
Journal: QIMS | Citations: 62
URL: https://qims.amegroups.com/article/viewFile/80770/pdf
Notes: Complete automated pipeline from raw EEG to classification -- practical reference for CS students

---

## Part 4: EEG Neurofeedback -- Intervention Direction

### 4.1 Neurofeedback Methodology Basics [MUST READ]

Enriquez-Geppert, Huster and Herrmann (2017)
Title: EEG-Neurofeedback as a Tool to Modulate Cognition and Behavior: A Review Tutorial
Journal: Frontiers in Human Neuroscience | Citations: 515
URL: https://www.frontiersin.org/articles/10.3389/fnhum.2017.00051/pdf

Core Content:
- Neurofeedback uses operant conditioning to help subjects learn to regulate their own brain activity
- Alpha neurofeedback (training to increase 8-12 Hz power) improves working memory and attention
- Theta/alpha ratio feedback enhances cognitive flexibility
- Relevance: AD/MCI patients have decreased alpha power --> restoring alpha via neurofeedback is the core hypothesis

---

Marzbani, Marateb and Mansourian (2016)
Title: Neurofeedback: A Comprehensive Review on System Design, Methodology and Clinical Applications
Journal: Basic and Clinical Neuroscience | Citations: 476
URL: http://bcn.iums.ac.ir/files/site1/user_files_c424bc/mansourian-A-10-739-1-d9cca26.pdf

System Design Reference:
- Typical neurofeedback flow: EEG acquisition -> real-time signal processing -> feature extraction -> feedback (visual/auditory) -> subject response
- Open-source toolchain: MNE-Python (preprocessing) + OpenViBE (real-time processing)
- Technical feasibility: existing hardware (e.g., OpenBCI) supports low-latency (<50ms) feedback

---

### 4.2 BCI Cognitive Rehabilitation Evidence

Lazarou et al. (2018)
Title: EEG-Based BCIs for Communication and Rehabilitation of People with Motor Impairment
Journal: Frontiers in Human Neuroscience | Citations: 334
URL: https://www.frontiersin.org/articles/10.3389/fnhum.2018.00014/pdf
Notes: Closed-loop feedback architecture directly transferable to cognitive rehabilitation; task design (active vs. passive BCI) impacts neuroplasticity

---

Mane, Chouhan and Guan (2020)
Title: BCI for stroke rehabilitation: motor and beyond
Journal: Journal of Neural Engineering | Citations: 450
URL: https://iopscience.iop.org/article/10.1088/1741-2552/aba162/pdf
Notes: BCI-driven neuroplasticity mechanisms provide theoretical basis for cognitive recovery; ERD/ERS paradigm analogous to cognitive state modulation

---

### 4.3 Aging and Cognitive Neurofeedback Evidence

Ranasinghe and Mapa (2024)
Title: Functional connectivity and cognitive decline: a review of rs-fMRI, EEG, MEG, and graph theory
Journal: Exploration of Medicine | Citations: 29
URL: https://www.explorationpub.com/uploads/Article/A1001256/1001256.pdf
Notes: Multi-modal neuroimaging evidence supports alpha connectivity enhancement correlating with cognitive improvement; supports alpha neurofeedback for MCI

---

## Part 5: Classification and Closed-loop System Technology

### 5.1 EEG Machine Learning Classification

Miltiadous et al. (2021)
Title: Alzheimer's Disease and Frontotemporal Dementia: A Robust Classification Method of EEG Signals
Journal: Diagnostics | Citations: 157
URL: https://www.mdpi.com/2075-4418/11/8/1437/pdf
Notes: Systematic comparison of SVM, KNN, random forest for AD EEG classification; recommended as Phase 1 classification benchmark

---

Chiarion et al. (2023)
Title: Connectivity Analysis in EEG Data: A Tutorial Review
Journal: Bioengineering | Citations: 177
URL: https://www.mdpi.com/2306-5354/10/3/372/pdf
Notes: Detailed tutorial on coherence, Granger causality, phase synchrony metrics

---

### 5.2 EEG-BCI Technology Survey

Gu et al. (2021)
Title: EEG-Based BCIs: A Survey of Recent Studies on Signal Sensing and Computational Intelligence
Journal: IEEE Transactions on Computational Biology | Citations: 379
URL: http://hdl.handle.net/10453/147196
Notes: Covers EEG hardware, processing algorithms (CSP, deep learning), and BCI applications; key reference for Phase 2 system design

---

Peksa and Mamchur (2023)
Title: State-of-the-Art on Brain-Computer Interface Technology
Journal: Sensors | Citations: 139
URL: https://www.mdpi.com/1424-8220/23/13/6001/pdf
Notes: Latest BCI review covering real-time processing latency, accuracy, and usability for non-invasive systems

---

## Part 6: Research Gaps and Innovation Positioning

| Research Gap | Current Status | This Project's Response |
|---|---|---|
| Limited reproducibility of SCD EEG biomarkers | Lassi 2023: single-center study | Validate with public multi-center datasets |
| Microstate + spectral features rarely combined | Most studies use one method only | Integrate microstate temporal params with alpha power |
| Systematic real-time EEG neurofeedback for MCI lacking | Most intervention studies use healthy subjects | Phase 2: closed-loop protocol designed for MCI |
| Biomarker-to-treatment closed-loop design gap | Diagnosis and intervention research separate | Propose detect-feedback-validate integrated framework |

---

## Part 7: Recommended Reading Pathway (by Priority)

### Priority 1 (Essential, directly supports Phase 1)
1. Lassi et al. (2023) -- Core microstate degradation paper (already downloaded)
2. Michel and Koenig (2018) -- Microstate methodology basis (already downloaded)
3. Meghdadi et al. (2021) -- EEG biomarker classification benchmark

### Priority 2 (Phase 1 analysis and methods)
4. Babiloni et al. (2019) -- AD electrophysiology mechanism review
5. Babiloni et al. (2021) -- AD clinical trial EEG expert consensus
6. Miltiadous et al. (2021) -- Classification method benchmark

### Priority 3 (Phase 2 intervention system design)
7. Enriquez-Geppert et al. (2017) -- Neurofeedback methodology tutorial
8. Marzbani et al. (2016) -- Neurofeedback system design review
9. Mane et al. (2020) -- BCI neuroplasticity mechanism

### Background Reading
10. Yuan and Zhao (2025) -- Latest qEEG review (2025)
11. Gu et al. (2021) -- EEG-BCI technology survey
12. Chiarion et al. (2023) -- EEG connectivity analysis tutorial

---

## Part 8: Two-Phase Research Framework

### Phase 1: EEG Biomarkers (aligned with current research proposal)
- Data: Public datasets (TUAB / OpenNeuro ds004504 / PhysioNet)
- Methods: Microstate degradation analysis + alpha/theta spectral features
- Output: HC/SCD/MCI classification model (logistic regression + SVM + random forest)
- Timeline: Weeks 1-20 (see research proposal timeline)
- Key references: Lassi 2023 / Meghdadi 2021 / Babiloni 2021

### Phase 2: Real-time Cognitive Feedback System (proposed extension)
- Signal: Real-time EEG -> MNE-Python streaming (LSL protocol)
- Features: Alpha peak frequency + microstate state label
- Feedback: Visual/auditory signals (gamified interface, e.g., Unity)
- Validation: Healthy subjects first (n>=10) -> controlled MCI subject trial
- Key references: Enriquez-Geppert 2017 / Marzbani 2016 / Mane 2020

---

## Part 9: Available Dataset Resources

| Dataset | Type | Diagnostic Labels | Access |
|---------|------|-------------------|--------|
| TUAB (Temple Univ.) | Resting EEG | Clinical diagnosis | https://www.isip.piconepress.com/projects/tuh_eeg/ |
| OpenNeuro ds004504 | Resting EEG | HC/MCI/AD | https://openneuro.org/datasets/ds004504 |
| PhysioNet | Multi-type EEG | Various | https://physionet.org |
| ADNI | Multi-modal | HC/MCI/AD | https://adni.loni.usc.edu |

---

## Part 10: Full Reference List

[1] Lassi M, et al. (2023). Degradation of EEG microstates patterns in SCD and MCI. NeuroImage: Clinical, 38, 103407. doi:10.1016/j.nicl.2023.103407
[2] Meghdadi AH, et al. (2021). Resting state EEG biomarkers of cognitive decline. PLoS ONE, 16(1), e0244180. doi:10.1371/journal.pone.0244180
[3] Babiloni C, et al. (2019). What electrophysiology tells us about Alzheimer's disease. Neurobiology of Aging, 85, 58-73. doi:10.1016/j.neurobiolaging.2019.09.008
[4] Babiloni C, et al. (2021). Measures of resting state EEG rhythms for clinical trials in AD. Alzheimer's and Dementia, 17(8), 1365-1382. doi:10.1002/alz.12311
[5] Michel CM, Koenig T. (2018). EEG microstates. NeuroImage, 180, 577-593.
[6] Enriquez-Geppert S, et al. (2017). EEG-Neurofeedback as a Tool to Modulate Cognition. Front Human Neuroscience, 11, 51. doi:10.3389/fnhum.2017.00051
[7] Marzbani H, et al. (2016). Neurofeedback: A Comprehensive Review. Basic Clinical Neuroscience, 7(2), 143-158.
[8] Mane R, et al. (2020). BCI for stroke rehabilitation: motor and beyond. J Neural Engineering, 17(4), 041003. doi:10.1088/1741-2552/aba162
[9] Miltiadous A, et al. (2021). AD and FTD: A Robust EEG Classification Method. Diagnostics, 11(8), 1437. doi:10.3390/diagnostics11081437
[10] Lazarou I, et al. (2018). EEG-Based BCIs for Rehabilitation. Front Human Neuroscience, 12, 14. doi:10.3389/fnhum.2018.00014
[11] Adebisi AT, et al. (2024). EEG Brain Functional Network for Dementia. IEEE TNSRE, 32, 1275-1284. doi:10.1109/tnsre.2024.3374651
[12] Prado P, et al. (2022). Dementia ConnEEGtome. Int J Psychophysiology, 172, 37-47. doi:10.1016/j.ijpsycho.2021.12.008
[13] Yuan Y, Zhao Y. (2025). qEEG biomarkers in AD and MCI. Front Aging Neuroscience, 17, 1522552. doi:10.3389/fnagi.2025.1522552
[14] Gu X, et al. (2021). EEG-Based BCIs Survey. IEEE TCBB, 18(5), 1645-1666. doi:10.1109/tcbb.2021.3052811
[15] Chiarion G, et al. (2023). EEG Connectivity Analysis Tutorial. Bioengineering, 10(3), 372. doi:10.3390/bioengineering10030372
[16] Luo Y, et al. (2020). Schizophrenia Biomarkers from EEG Microstates. IEEE Access, 8, 213961. doi:10.1109/access.2020.3037658
[17] Ding Y, et al. (2021). Automated AD discrimination using resting EEG. QIMS, 11(12), 4960. doi:10.21037/qims-21-430
[18] Ranasinghe PHKIS, Mapa MST. (2024). Functional connectivity and cognitive decline. Exploration of Medicine, 5, 256-278. doi:10.37349/emed.2024.00256

---
Note: This literature review was compiled with AI assistance. All references have been verified through academic databases.
Recommendation: Read each original paper before submitting to your supervisor.