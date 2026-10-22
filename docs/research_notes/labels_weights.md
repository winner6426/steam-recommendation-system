# Labels and weights from Steam reviews: explicit negatives, playtime confidence, helpful/funny votes

Context: Steam review dataset with `is_recommended` True/False, `hours` 0–1000 (capped, median 27.3). The decisions to inform are (1) whether to use `is_recommended=False` as explicit negatives in MF (weighted BCE / BPR) and InfoNCE, (2) whether to use `w = 1 + α·log(1+hours)` as a confidence weight, and (3) whether to use helpful/funny counts as weights.

## Q1. Hu, Koren & Volinsky (ICDM 2008): the exact confidence formulas

### Takeaway
Verified from the original PDF. The paper binarizes preference (`p_ui = 1 if r_ui > 0`, else 0) and uses confidence `c_ui = 1 + α·r_ui` with α = 40 in experiments, or the log variant `c_ui = 1 + α·log(1 + r_ui/ε)` (Eq. 6). On the TV data they used Eq. 6 with ε = 10⁻⁸. The model has no explicit-negative label. All unobserved pairs get the same uniform low confidence, and the authors name "splitting those zero values into different confidence levels" as future work.

### Cited Findings
- Preference is binarized: `p_ui = 1` if `r_ui > 0`, `p_ui = 0` if `r_ui = 0`. Zero values carry low confidence because "not taking any positive action on an item can stem from many other reasons beyond not liking it". — [Hu, Koren, Volinsky 2008, Sec. 4 (PDF)](http://yifanhu.net/PUB/cf.pdf)
- Linear confidence: "A plausible choice for c_ui would be: c_ui = 1 + α r_ui … In our experiments, setting α = 40 was found to produce good results." — [Hu et al. 2008](http://yifanhu.net/PUB/cf.pdf)
- Log variant: "One alternative method that also worked well to us is setting c_ui = 1 + α log(1 + r_ui/ε). (6)". The paper also says "one can derive p_ui differently from r_ui, by setting a minimum threshold on r_ui for the corresponding p_ui to be non-zero." — [Hu et al. 2008](http://yifanhu.net/PUB/cf.pdf)
- On the TV data, r_ui ranged from near 0 (channel flipping) to hundreds (DVR recording for hours a day), so "we employ the log scaling scheme (6) with ε = 10⁻⁸". In the test set they set entries with r_ui < 0.5 to zero, because "watching less than half of a program is not a strong indication that a user likes the program". They also down-weighted later shows in long single-channel sessions, since the TV may have been left on. — [Hu et al. 2008, Sec. 6](http://yifanhu.net/PUB/cf.pdf)
- Limitation the authors state: "we decided to treat all user-item pairs associated with a zero preference with the same uniform confidence level … a more careful analysis would split those zero values into different confidence levels". — [Hu et al. 2008, Conclusion](http://yifanhu.net/PUB/cf.pdf)
- Secondary sources sometimes write the log form without ε, or cite α = 40 and ε ≈ 1e-8 as defaults. The original paper is authoritative. — [AIWiki WALS (secondary)](https://aiwiki.ai/wiki/weighted_alternating_least_squares_wals/raw); [Uncertainty in Repeated Implicit Feedback, arXiv 2505.02492](https://arxiv.org/pdf/2505.02492)

### Inferences
- The proposed `w = 1 + α·log(1+hours)` is Hu's Eq. 6 with ε = 1 hour. That is a reasonable unit choice. Hu's ε = 10⁻⁸ with r in show units makes the log extremely steep near 0, and α = 40 was tuned for the linear form on TV data, so neither should be copied. Tune α on validation (T1 data only), including α = 0 (unweighted) as the control.
- Hu's model has no slot for "observed but disliked". In Steam, a review with `is_recommended=False` and many hours would get *high* confidence on p=1 under pure Hu. That is wrong for this dataset, so hours confidence should multiply the label actually observed (positive or explicit negative), not imply positivity.

### Gaps
- The α = 40 value applies to the linear form. The paper does not report the α used with Eq. 6 (not found in the text I extracted).

## Q2. Incorporating explicit negative feedback into implicit CF / BPR / softmax

### Takeaway
Several peer-reviewed works report gains from treating explicit negatives (dislikes, skips, low ratings) as a separate class: graded or multi-level BPR, sign-aware GNN BPR, and a YouTube "not-to-recommend" loss with live A/B gains. YouTube found that a dislike used only as an input feature had no significant effect. It had to be a *training label*.

### Cited Findings
- Original BPR assumes only positive observations. Unobserved pairs mix "true negatives" with "missing values". — [Rendle et al., UAI 2009](https://arxiv.org/pdf/1205.2618)
- **YouTube (RecSys 2023 industry track)**: a "not-to-recommend" loss `L_i = −log(1 − p(y_i|s_i))` for negative-labelled items, added to the softmax positive loss with per-label weights: `L = −Σ_{pos} r_i·log p(y_i|s_i) − Σ_{neg} w_i·log(1 − p(y_i|s_i))`. The authors chose it over (a) negative-valued label weights in cross-entropy, which has "gradient blow-up when p→0", and (b) "include and upweight negative feedback items among the softmax negative samples". The decoupled loss gives a finite gradient and "more targeted learning". Weights "can be tuned according to the feedback strength, signal density, and the relative loss value magnitude compared to positive labels." — [Wang et al., Learning from Negative User Feedback…, arXiv 2308.12256](https://arxiv.org/abs/2308.12256)
- YouTube live results: dislikes used as both input feature and training label reduced the dislike rate by 2.44%. Dislikes as an input feature only gave −0.34% (not significant). The heuristic of excluding disliked items from the input sequence gave −0.84%. Repeated dislikes on the same creator fell 9.60%. Skips as negative labels gave +0.40% overall enjoyment and −1.44% repeated skips from the same creator. — [arXiv 2308.12256](https://arxiv.org/abs/2308.12256)
- **Skips as InfoNCE negatives (music)**: an InfoNCE term with the next non-skipped track as positive and *skipped tracks in the session* as the noise set, `L = α·L_NCE + β·L_NLL` (α = β = 0.5 chosen from {0.25, 0.5, 0.75, 1}). Only skip strengths 1–2 ("played very briefly/briefly") count as negatives. Reported HR@1 gains of +8% to +20%, shrinking at larger K (one config at HR@20 was −1.3%). No significance tests reported. — [Seshadri & Knees, arXiv 2309.11623](https://arxiv.org/html/2309.11623v2); follow-up: [Seshadri, Shashaani, Knees, arXiv 2409.07367](https://arxiv.org/pdf/2409.07367)
- **Graded BPR (BPR++)**: derives extra pairwise preferences from graded "importance" information and biases optimization toward them. Measurable accuracy gains over BPR on e-commerce data. — [Lerche & Jannach, RecSys 2014 (PDF)](https://web-ainf.aau.at/pub/jannach/files/Conference_RecSys2014a.pdf)
- **Multi-channel / multi-level BPR**: feedback sources are ordered into "levels", and negatives are sampled from lower levels. A parameter β sets the share of unobserved items among negatives. The best sampler depends on dataset properties such as popularity skew (details from abstract and secondary summaries; full PDF not read). — [Loni et al., RecSys 2016](https://research.tudelft.nl/en/publications/bayesian-personalized-ranking-with-multi-channel-user-feedback); [code](https://github.com/mkurovski/multi_channel_bpr)
- **SiReN (IEEE TNNLS 2022)**: splits a signed bipartite graph into positive and negative edges (low ratings = negative), with a GNN on positives, an MLP on negatives, attention fusion, and a sign-aware BPR loss. Reported to beat embedding-based baselines. — [Seo et al., arXiv 2108.08735](https://arxiv.org/pdf/2108.08735)
- On Steam specifically, a non-peer-reviewed student project reported that adding explicit recommend/not-recommend "doesn't change the performance much" in ALS. — [Batra et al., arXiv 2305.04890](https://arxiv.org/pdf/2305.04890) (summary via search snippet; I did not verify details in the full PDF)

### Inferences
- For the project's MF + weighted BCE: give `False` reviews label 0 with its own weight `w_neg`, separate from the weight on sampled unknowns. This is Hu's suggested "different confidence levels for zeros" plus YouTube's tunable negative-label weight.
- For BPR: use graded pairs (pos ≻ unknown and pos ≻ explicit-neg, with optional unknown ≻ explicit-neg as in BPR++/MC-BPR levels). Mixing explicit negatives and unknowns via a ratio parameter (MC-BPR's β) matches the project's sampler, "explicit False + random unknown".
- For InfoNCE: two evidence-backed options are (a) add explicit negatives to the softmax denominator, possibly upweighted (the alternative YouTube mentions; the Seshadri approach), and (b) a separate `−log(1−p)` term (YouTube). Option (b) decouples the gradients. Either one needs a matched Hybrid+BPR control with the same negatives, per AGENTS.md.
- In this dataset an explicit negative is still evidence the user *chose to buy/play* the game. It is a "consumed but disliked" signal, not "never interested". The effect on *ranking* (which full-catalog metrics target) may therefore be smaller than YouTube's effect on dislike rate. Running a no-explicit-neg ablation is warranted.

### Gaps
- I did not find a paper that evaluates explicit negatives specifically on Steam review data under a temporal split.
- Exact weights YouTube used for dislikes and skips are not disclosed.
- I did not read the MC-BPR full text. The specific numbers are unverified.

## Q3. Playtime / dwell / watch-time weighting: does it help, and what are the biases?

### Takeaway
Engagement-magnitude weighting is standard: Hu's confidence, and YouTube's weighted logistic regression with positives weighted by watch time. But duration is a confounder. Longer items accumulate more engagement regardless of preference, and Zhan et al. (KDD 2022) show that this biases models toward long items. The Steam analogue is a "long-game bias": MMOs and sandbox games accrue hundreds of hours. A per-game normalized (quantile) hours signal is the debiased analogue.

### Cited Findings
- YouTube ranking used weighted logistic regression: positive (clicked) impressions are weighted by watch time and negatives get unit weight, so the learned odds approximate expected watch time. — [Covington et al., RecSys 2016 (PDF)](https://www.gwern.net/doc/ai/nn/retrieval/2016-covington.pdf); [summary](https://blog.acolyer.org/2016/09/19/deep-neural-networks-for-youtube-recommendations/)
- **Duration bias**: watch time depends on video duration, not only on interest. Duration is a confounder affecting both exposure and watch time. Models trained on raw watch time over-recommend long videos. The D2Q fix converts watch-time labels into *duration-dependent quantiles* (bucket by duration, then quantile within bucket). It outperformed VR and WLR baselines offline and in Kuaishou live A/B tests. — [Zhan et al., KDD 2022, arXiv 2206.06003](https://arxiv.org/abs/2206.06003)
- Hu et al. themselves note confounded engagement: long single-channel sessions where "the television might simply have been left on". They down-weighted those cases. — [Hu et al. 2008](http://yifanhu.net/PUB/cf.pdf)
- YouTube encodes skips "as part of the dwell time feature". Short engagement is treated as a negative-ish signal, not just a low-confidence positive. — [arXiv 2308.12256](https://arxiv.org/abs/2308.12256)
- Seshadri & Knees treat "played very briefly/briefly" as negatives and "played mostly" as not negative. This is a thresholding of engagement magnitude. — [arXiv 2309.11623](https://arxiv.org/html/2309.11623v2)
- Steam playtime has been used as implicit feedback for ALS in a student project. An unreviewed preprint reports up to 54.3% relative NDCG@20 gain from adaptive confidence weighting on "highly skewed Steam" data (an unreviewed preprint, flagged as weak evidence). — [Batra et al., arXiv 2305.04890](https://arxiv.org/pdf/2305.04890); [Adaptive Implicit Feedback Weighting preprint](https://sciety.org/articles/activity/10.21203/rs.3.rs-9216103/v1)
- Treating the reliability of repeated implicit feedback as uncertainty-based confidence in ALS improved results on two (non-Steam) datasets. — [arXiv 2505.02492](https://arxiv.org/pdf/2505.02492); [Deezer write-up](https://newsroom-deezer.com/2025/06/umap-sguerra-html/)

### Inferences
- `1 + α·log(1+hours)` with a 1000 h cap is a defensible Hu-style weight. The log compresses the heavy tail, which matches the motivation for Hu's Eq. 6. Expected risk: long-game bias (an analogue of duration bias). Games designed for long play would get systematically larger weights and could dominate the item factors. A D2Q-style alternative is the per-game percentile of hours, `w = 1 + α·q_game(hours)`, with quantiles fit on reviews before the cutoff only (protocol compliance).
- Hours should weight *both* positive and explicit-negative labels, as "confidence in the observed label". It should not convert hours into positivity. A `False` review after 300 h is a confident negative about the game's current state, or possibly a complaint about an update. This interpretation is my inference, not from a paper.
- The hours on a Steam review are likely the playtime *at review time* (or snapshot time). If they are snapshot hours, they include play after the review timestamp, which would be future leakage under the two-snapshot protocol. This needs checking against the dataset documentation and is **unverified**.
- Because the evaluation is full-catalog Recall/NDCG on binary held-out positives, weighting changes training emphasis but not the target. Gains are not guaranteed, so compare α = 0 vs α > 0 at an equal training budget.

### Gaps
- No peer-reviewed study found that quantifies long-game bias on Steam or tests per-game hours normalization.
- Whether `hours` in this specific dataset is at-review or at-snapshot is not established by the sources gathered.

## Q4. Helpful/funny vote counts as sample weights in CF

### Takeaway
I found no CF/implicit-feedback paper that uses helpful/funny counts as *interaction sample weights*. Helpfulness appears in review-based (text) recommenders, as a review-quality filter or attention weight. Votes are known to be biased (winner-circle, early-bird), and they accumulate after the review is posted. Using them as weights would leak future information and conflate reader usefulness with reviewer preference strength.

### Cited Findings
- Helpfulness-aware review-based neural recommendation: helpfulness is predicted for unvoted reviews and used to weight review *text*. Reported improvements on four benchmarks. This is review-text modeling, not interaction weighting. — [Helpfulness-aware Review Based Neural Recommendation](https://bohrium.dp.tech/paper/arxiv/5f4d8e259fced0a24b14444c)
- Mauro, Ardissono & Petrone (IP&M 2020) estimate helpfulness from deviation features instead of votes and use it to select higher-quality data for a collaborative recommender. — [IP&M 2020 (PDF)](https://iris.unito.it/retrieve/e27ce430-e3b1-2581-e053-d805fe0acbaa/Information_Processing_MGMT__2020.pdf)
- Helpful votes are sparse ("not commonly available" per review). The thesis estimates helpfulness with weak supervision instead. — [Xi Wang PhD thesis, Glasgow 2022](https://theses.gla.ac.uk/82979/1/2022XiWangPhD.pdf)
- Vote biases: "imbalance vote bias, the winner circle bias, and the early bird bias" (patent text; not peer-reviewed). — [US patent 8930366](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/8930366)
- User/item-aware helpfulness estimation (arXiv 2011.10456) shows that helpfulness depends on reviewer and item context, not only content. — [arXiv 2011.10456](https://arxiv.org/pdf/2011.10456.pdf)

### Inferences
- Votes are cast by *other* users after the review is posted. Counts at the snapshot time therefore include post-cutoff activity, which violates the temporal protocol. The project's AGENTS.md already lists `helpful` and `funny` among future-derived features to avoid.
- Votes measure how useful a review was to readers, so they do not show how strongly the reviewer likes the game. "Funny" votes in particular often go to joke or negative reviews. Weighting by them would also amplify popularity and early-bird effects. Recommendation: do not use them as weights. At most, they could support an exploratory analysis with an explicit leakage caveat.

### Gaps
- No direct empirical study found of helpful-vote weighting in implicit CF, positive or negative. Absence of evidence is the finding.

## Q5. Weighted negatives in InfoNCE / sampled softmax

### Takeaway
There are three relevant mechanisms: (a) logQ correction for popularity-skewed in-batch negatives (Yi et al. 2019), (b) importance-reweighted "hard" negatives (Robinson et al. ICLR 2021), and (c) explicit negatives added to the denominator or a separate `−log(1−p)` term (YouTube 2023; Seshadri & Knees 2023). For a fair Phase 3 vs Hybrid+BPR comparison, the negative source and weighting should be matched.

### Cited Findings
- logQ correction: each logit is corrected by subtracting the log of the item's sampling probability. In-batch negatives follow a power law, so without the correction popular items are over-penalized and rare items are under-discriminated. This is now standard in industrial two-tower retrieval. — [Yi et al., RecSys 2019 (notes)](https://charleslow.github.io/notebook/book/papers/yi_2019.html); [Correcting the LogQ Correction, arXiv 2507.09331](https://arxiv.org/abs/2507.09331)
- A 2025 follow-up argues standard logQ mishandles the positive item, which is always in the denominator. It proposes a corrected form with consistent gains. — [arXiv 2507.09331](https://arxiv.org/abs/2507.09331)
- Hard-negative InfoNCE: a reweighted negative distribution with a hardness parameter β, implemented as importance sampling at zero extra cost, with debiasing to avoid false hard negatives. Gains reported on image/graph/text tasks (not recsys). — [Robinson et al., ICLR 2021, arXiv 2010.04592](https://arxiv.org/abs/2010.04592v2); [code](https://www.github.com/joshr17/HCL)
- Upweighting negative-feedback items among softmax negatives is a recognized alternative. YouTube preferred a decoupled `−w·log(1−p)` term for more targeted gradients. — [arXiv 2308.12256](https://arxiv.org/abs/2308.12256)
- Explicit negatives (skips) as the InfoNCE noise set, combined with the main loss through tuned weights. — [arXiv 2309.11623](https://arxiv.org/html/2309.11623v2)

### Inferences
- A practical Phase 3 design: in-batch or random-unknown negatives with logQ correction, plus the user's own `False` games appended to the denominator with weight `w_neg`, or a separate `−w_neg·log(1−p)` term. The Hybrid+BPR control should use the same negative pool (explicit False + random unknowns) so that the comparison isolates the loss.
- Hard-negative reweighting (Robinson) is close to the project's "semantic semi-hard" ablation. It should stay an ablation, not the main sampler.
- Note that false negatives are common among unknowns (users simply unaware of games). Debiased or hard-negative weighting partially addresses this, but none of these sources tests it on Steam.

### Gaps
- I did not read the original Yi et al. 2019 PDF. Its description here comes from secondary notes plus the 2025 follow-up.
- I did not find Chuang et al. (Debiased Contrastive Learning, NeurIPS 2020) through search in this session. It is referenced only indirectly via the HCL code. Unverified.
- No source found that compares "explicit negatives in the denominator" vs a "separate not-to-recommend term" head-to-head on public data.
