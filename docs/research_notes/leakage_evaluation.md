# Leakage from aggregated/snapshot features and temporal evaluation protocols in recommender systems

Scope: evidence for deciding whether scrape-time aggregated fields (positive_ratio, rating label, user_reviews, helpful/funny, users.products/users.reviews, current price/discount, steam_deck) can be features/weights when training up to cutoff T and evaluating later; how to use review `date` and release date; full-catalog vs sampled evaluation; snapshot content metadata.

Method note: primary texts were read for Kaufman et al. (KDD 2011 PDF), Cañamares & Castells (RecSys 2020 PDF), Sun (SIGIR 2023, ar5iv HTML), Hidasi & Czapp (RecSys 2023, ar5iv HTML) and the Ji et al. TOIS abstract. Meng et al., Campos et al., Krichene & Rendle and Ji et al. SIGIR 2020 were checked only through abstracts and index pages (marked "[abstract-level]").

## Q1. Definitions and examples of leakage, recsys-specific leakage, and split strategies

### Takeaway
Leakage means using information that would not legitimately be available when the prediction is made. The "no-time-machine" rule says a feature must be observed before its target. Recsys studies show that ignoring the global timeline (random splits, leave-one-out) leaks future data and can reorder model rankings unpredictably. A global temporal split at an absolute cutoff is the recommended remedy.

### Cited Findings
- Kaufman, Rosset & Perlich define leakage as "the introduction of information about the data mining target, which should not be legitimately available to mine from". They note that leakage "at the very least ... leads to overestimation of the model's performance" — [Kaufman et al., KDD 2011 (PDF)](https://www.cs.umb.edu/~ding/history/470_670_fall_2011/papers/cs670_Tran_PreferredPaper_LeakingInDataMining.pdf); bibliographic record [dblp](https://dblp.org/rec/conf/kdd/KaufmanRP11.html); TKDD 2012 journal version (adds Stitelman) [DOI 10.1145/2382577.2382579](https://www.doi.org/10.1145/2382577.2382579)
- The "no-time-machine requirement": "a legitimate model only build[s] on features with information from a time earlier (or sometimes, no later) than that of the target" — [Kaufman et al.](https://www.cs.umb.edu/~ding/history/470_670_fall_2011/papers/cs670_Tran_PreferredPaper_LeakingInDataMining.pdf)
- The avoidance method has two parts: legitimacy tags, which are timestamps in the temporal case, and a "learn-predict separation", which is "a cut at some point in time that segments training from evaluation examples". The paper warns that updates to database records are usually not timestamped, so whole records end up with one timestamp, "and this loss of information is often the source of all evil that allows leakage" — [Kaufman et al.](https://www.cs.umb.edu/~ding/history/470_670_fall_2011/papers/cs670_Tran_PreferredPaper_LeakingInDataMining.pdf)
- Two kinds of leakage: (a) leakage in features and (b) leakage in training examples. Leakage can also enter through design decisions, namely "selecting or designing features", "algorithm or parametric model selection" and "meta-parameter value choices" informed by data beyond the separation point. The authors call this form "perhaps the most dangerous" — [Kaufman et al.](https://www.cs.umb.edu/~ding/history/470_670_fall_2011/papers/cs670_Tran_PreferredPaper_LeakingInDataMining.pdf)
- Examples relevant to aggregated fields:
  - Kohavi's "session length" attribute was added to every page-view record at session end and leaked the target. The fix was "page number in session", meaning the count up to the prediction point.
  - In the IBM "Websphere" case, website text was sampled after the prospect had already become a customer.
  - Using "current (2007) IMDB data" that is "not rolled-back to 2005" for the 2007 KDD Cup Netflix task "is simple leakage".
  - Source: [Kaufman et al.](https://www.cs.umb.edu/~ding/history/470_670_fall_2011/papers/cs670_Tran_PreferredPaper_LeakingInDataMining.pdf)
- Ji, Sun, Zhang & Li (TOIS 2023): "Data leakage is caused by not observing global timeline in evaluating recommenders". With leave-last-one-out on MovieLens-25M, Yelp, Amazon-music and Amazon-electronic:
  - BPR, NeuMF, SASRec and LightGCN all recommend items that are future relative to the test instance.
  - Leakage "does impact models' recommendation accuracy".
  - Relative performance orders "become unpredictable with different amount of leaked future data in training".
  - The authors propose a timeline scheme.
  - Sources: [arXiv 2010.11060](https://arxiv.org/abs/2010.11060v4); [dblp TOIS 41(3) 75:1–75:27](https://dblp.org/rec/journals/tois/JiS0L23)
- Sun (SIGIR 2023) describes leakage as "accessing future data that is impossible to access in reality". The paper proposes a timeline scheme: order all interactions chronologically, split the timeline into windows W, and evaluate window by window. The model may learn from the current window and all earlier ones, so "the amount of data leakage is controlled by the size of time window W" — [Sun, arXiv 2210.04149 (ar5iv)](https://ar5iv.labs.arxiv.org/html/2210.04149); [dblp SIGIR 2023](https://dblp.uni-trier.de/rec/conf/sigir/Sun23.html)
- Sun's case study of 88 RecSys 2020–2022 papers:
  - random split: 30 papers (34.1%)
  - leave-one-out: 22 (25.0%)
  - single time point: 17 (19.5%)
  - simulation-based online: 15 (17.0%)
  - sliding window: 4 (4.5%)
  - "59.1% ... follow" the two settings with no global timeline (random split and leave-one-out); 26% use Popularity as a baseline
  - Source: [Sun (ar5iv)](https://ar5iv.labs.arxiv.org/html/2210.04149)
- Sun notes that MovieLens rating timestamps do not reflect the time of the actual interaction (watching), so a time split can still be imperfect. This is a caveat on what the timestamp means — summarized in [Sun, arXiv PDF](https://arxiv.org/pdf/2210.04149) (seen in a search summary; not re-verified in the full text by me)
- Meng, McCreadie, Macdonald & Ounis (RecSys 2020) [abstract-level]:
  - They compared leave-one-last-item, leave-one-last-basket and global temporal split, using 7 models on 2 datasets.
  - Split strategies "strongly impact the ranking of systems under the same dataset and metrics".
  - Recommendations: report the split, report performance under global temporal splitting, and release the splits.
  - Sources: [arXiv 2007.13237](https://arxiv.org/pdf/2007.13237); [Glasgow eprints](https://eprints.gla.ac.uk/222234)
- Campos, Díez & Cantador (UMUAI 24:67–119, 2014) [abstract-level]: contradictory results on time-aware recommenders are traced to divergent evaluation protocols, meaning the metrics and methodologies used. Different evaluation conditions "yield to remarkably distinct performance and relative ranking values". The paper proposes a protocol description framework — [UNED portal record](https://portalcientifico.uned.es/documentos/5d39994c2999520684440e57?lang=en); [researchr](https://researchr.org/publication/CamposDC14)
- Hidasi & Czapp (RecSys 2023): "information leaking through time" is one of four widespread flaws.
  - Random splits are appropriate only if preferences are static, which is "not the case" given catalog changes, shifting interests and promotions.
  - Under leave-one-out, sequences "end at different times, thus the two sets overlap in time", which makes the task easier and unrealistic.
  - Source: [Hidasi & Czapp (ar5iv)](https://ar5iv.labs.arxiv.org/html/2307.14951)
- Hidasi & Czapp also warn against "overzealous preprocessing". Filtering changes the meaning and comparability of results, so they advise "making only the most necessary preprocessing steps" — [Hidasi & Czapp](https://ar5iv.labs.arxiv.org/html/2307.14951)

### Inferences
- Under Kaufman's no-time-machine rule, any field that aggregates reviews posted after cutoff T is an illegitimate feature for targets after T. This covers positive_ratio, the rating summary label, user_reviews, helpful/funny, and users.reviews/users.products as scraped. Recomputing the same quantity from reviews dated ≤ T (Kohavi's "page number in session" fix) is the legitimate substitute.
- Using such fields for hyperparameter or feature selection is Kaufman's "leakage by design decisions", even if they are excluded from the final model.
- Current price/discount and steam_deck are the same type of case as "current IMDB data not rolled back": a value observed at scrape time, not at review time. They are illegitimate for evaluating past interactions unless historical values are available.
- A global temporal split (T1 for validation, T2 for test) matches Meng et al.'s recommendation and Kaufman's learn-predict separation. Sun's windowed timeline scheme is the stricter generalization.

### Gaps
- Meng et al. and Campos et al. full texts were not read. Their specific datasets, metrics and magnitudes are unverified.
- None of these sources directly analyze Steam data, review-vote counts (helpful/funny) or platform flags. Their applicability here is by inference.

## Q2. Popularity features and popularity bias: is item popularity computed from future data a known leakage source?

### Takeaway
Yes. Sun (SIGIR 2023) explicitly calls recommending "items that are popular in future by using the frequency counting that happened in future" unrealistic. Ji et al. (SIGIR 2020) show that how popularity is timed materially changes the popularity baseline's measured performance. Popularity is also a strong confounder in offline evaluation (Cañamares & Castells 2018).

### Cited Findings
- Sun: popularity is usually counted over all training interactions, with no time window and no updates along the timeline. Under leave-one-out, "the popularity method may recommend some items to u1 that are very popular in future". Also: "Clearly, recommending items that are popular in future by using the frequency counting that happened in future is unrealistic." — [Sun (ar5iv)](https://ar5iv.labs.arxiv.org/html/2210.04149)
- Ji, Sun, Zhang & Li (SIGIR 2020) [abstract-level]:
  - They reviewed 12 papers and 6 toolkits; MostPop is commonly computed as interaction counts over the training data.
  - On MovieLens, measuring popularity at the time the user interacts improves the baseline "by 70% or more".
  - They propose RecentPop and DecayPop.
  - Sources: [arXiv 2005.13829](https://arxiv.org/abs/2005.13829); [ar5iv](https://ar5iv.labs.arxiv.org/html/2005.13829)
- Cañamares & Castells (SIGIR 2018) [abstract-level]: the paper analyzes when popularity is an effective signal, and cases where following popularity can perform worse than random. Its crowdsourced unbiased dataset shows contradictions between accuracy in a biased offline setting and accuracy estimated from unbiased observations — [Cañamares & Castells 2018 PDF](https://castells.github.io/papers/sigir2018.pdf); [code](https://github.com/ir-uam/SIGIR2018)
- Kaufman's KDD Cup 2007 example ("How Many Reviews") shows that future review counts of related titles share causal ancestors (promotions, trends) with the target. That makes them leaky even when the titles are disjoint — [Kaufman et al.](https://www.cs.umb.edu/~ding/history/470_670_fall_2011/papers/cs670_Tran_PreferredPaper_LeakingInDataMining.pdf)

### Inferences
- user_reviews (count) and positive_ratio are the game's final popularity and quality, aggregated over the whole crawl period. They are therefore future popularity in Sun's sense when the evaluation targets come after T.
  - Such fields can inflate scores for both a popularity baseline and any model that consumes them as features or sample weights.
  - Because they correlate with test-period positives, they also act as a label proxy. This matches Kaufman's "proxy for the label" type of leakage.
- Legitimate alternatives: popularity, positive rate and review count computed only from reviews dated ≤ T, optionally time-decayed in the spirit of RecentPop/DecayPop. The same applies to popularity-based negative samplers.
- Steam's "rating" summary label is a deterministic function of the positive ratio and the review count. The same reasoning therefore applies. (This is an inference about Steam's labelling. I did not verify Steam's exact rule from a primary source.)

### Gaps
- I found no paper that quantifies leakage from item-level aggregate side features (as opposed to the popularity baseline) in a recsys benchmark. The claim that such features inflate hybrid models is reasoned from the Kaufman, Sun and Ji findings, not directly measured.
- No source found on helpful/funny votes specifically.

## Q3. Full-catalog vs sampled evaluation

### Takeaway
Sampled metrics (ranking the target against a few random negatives) are inconsistent with exact metrics and can reverse model comparisons. Krichene & Rendle recommend avoiding sampling. Cañamares & Castells add a nuance: tiny target sets are badly distorted, and full sets may not be the perfect reference either. Full-catalog ranking remains the standard reference, and the setting should be reported explicitly.

### Cited Findings
- Krichene & Rendle (KDD 2020, pp. 1748–1757) [abstract-level]:
  - Sampled metrics "do not persist relative statements, e.g., 'algorithm A is better than B', not even in expectation".
  - With small sample sizes, metrics collapse toward AUC.
  - Corrections (minimizing bias/MSE) can help.
  - "Sampling should be avoided for metric calculation."
  - Sources: [Google Research page](https://research.google/pubs/on-sampled-metrics-for-item-recommendation/); [IJCAI 2021 extended abstract](https://www.ijcai.org/proceedings/2021/651)
  - Note: the arXiv ID [1912.02263](https://arxiv.org/pdf/1912.02263) was described in one search summary as a Rendle companion report. I did not verify whether it is the Krichene & Rendle KDD preprint or Rendle's separate "Evaluation Metrics for Item Recommendation under Sampling".
- Cañamares & Castells (RecSys 2020):
  - "comparative evaluation using reduced target sets contradicts in many cases the corresponding outcome using large targets".
  - Comparing against unbiased evaluation, "minimum target sets incur substantial distortion in pairwise system comparisons, while maximum sets may not be ideal either, and better options may lie in between".
  - Hyperparameters tuned for full targets are suboptimal under test-only targets, and vice versa.
  - They recommend reporting tie ratios alongside p-values.
  - Source: [Cañamares & Castells 2020 PDF](https://castells.github.io/papers/recsys2020.pdf); DOI 10.1145/3383313.3412259
- Cañamares & Castells exclude items with a training rating from every target set ("algorithms are not requested to recommend items that have a training rating"). They treat explicitly rated test items and unrated items as distinct target categories — [Cañamares & Castells 2020](https://castells.github.io/papers/recsys2020.pdf)
- Hidasi & Czapp: negative sampling at test time inflates accuracy and can reverse rankings.
  - Coveo: model A beats B by 8.1% in recall@20 under full ranking but trails by 4.7% with 100 samples.
  - Rees46: the MRR order A > C ≫ B becomes B > A ≫ C with 100 samples.
  - A target ranked 1,490th out of 10,000 items still has a ">90%" chance of reaching the top 20 with 100 samples.
  - Source: [Hidasi & Czapp (ar5iv)](https://ar5iv.labs.arxiv.org/html/2307.14951)

### Inferences
- Main results should rank the full eligible catalog at each cutoff, minus items the user already reviewed before the cutoff, consistent with the Cañamares & Castells exclusion convention. Any sampled evaluation should be labelled as secondary.
- The eligible catalog at cutoff T should contain only items that exist at T, for example release date ≤ T and/or at least one review ≤ T. Otherwise the candidate set includes "future items", which Ji et al. identify as a direct symptom of leakage. Release date is the natural legitimacy tag for item availability, in Kaufman's sense.
- Because Cañamares & Castells show that target-set choice interacts with popularity bias and with hyperparameter tuning, the same candidate definition must be used for validation (T1) and test (T2).

### Gaps
- Krichene & Rendle full text not read. The exact theorems and correction formulas are unverified here.
- No source found that gives a rule for choosing an "intermediate" target size. Cañamares & Castells only suggest that one may exist.

## Q4. Snapshot content metadata (description/tags as of scrape): how is this limitation acknowledged?

### Takeaway
I found no recsys paper that formally treats "current-snapshot item text/tags" as a leakage category. The closest authoritative precedents are Kaufman et al.'s examples: website text sampled after the outcome (IBM Websphere) and external data "not rolled-back" (current IMDB data). Both are explicitly called leakage. The defensible practice is to state the limitation, and to prefer fields that are plausibly static (title, developer, release date, base description) over volatile ones (user-applied tags, price, discount, platform flags).

### Cited Findings
- IBM Websphere: website content used for training "had been sampled at the point in time where the potential customer has already become a customer", and so contained traces of the target — [Kaufman et al.](https://www.cs.umb.edu/~ding/history/470_670_fall_2011/papers/cs670_Tran_PreferredPaper_LeakingInDataMining.pdf)
- "The use of external sources that are not rolled-back to 2005, such as using current (2007) IMDB data, is simple leakage" — [Kaufman et al.](https://www.cs.umb.edu/~ding/history/470_670_fall_2011/papers/cs670_Tran_PreferredPaper_LeakingInDataMining.pdf)
- Kaufman et al. note that records usually carry a single timestamp and lose their update history. This is exactly the snapshot situation — [Kaufman et al.](https://www.cs.umb.edu/~ding/history/470_670_fall_2011/papers/cs670_Tran_PreferredPaper_LeakingInDataMining.pdf)
- When historical legitimacy tags cannot be recovered, Kaufman et al. say "it may be possible to partly plug the leak but not to seal it completely". In the last-resort scenario, one "has to make do with that which is available" — [Kaufman et al.](https://www.cs.umb.edu/~ding/history/470_670_fall_2011/papers/cs670_Tran_PreferredPaper_LeakingInDataMining.pdf)
- Sun notes that side information (attributes) can ease sparsity, but does not discuss whether attributes may themselves leak future information — [Sun (ar5iv)](https://ar5iv.labs.arxiv.org/html/2210.04149)
- Dataset context: the Kaggle "Game Recommendations on Steam" dataset (Anton Kozyriev) is described by secondary sources as containing over 41M cleaned user recommendations. One project lists a games.csv steam_deck Boolean ("supported on Steam Deck") — [ecosyste.ms project listing](https://awesome.ecosyste.ms/projects/github.com%2Fabdxdev%2Fgame-recommendation-system); [Baselight mirror](https://baselight.app/u/kaggle/dataset/antonkozyriev_game_recommendations_on_steam). UNVERIFIED: the Kaggle page did not render, so the official scrape date and per-column definitions could not be confirmed.

### Inferences
- Description and tags as of scrape are a known, non-removable limitation (Kaufman's "make do" scenario). Papers and reports should state it explicitly and avoid claiming strict temporal purity for content features.
  - User-applied Steam tags plausibly evolve with the community response after release, which makes them partially outcome-dependent (a soft analogue of the Websphere case).
  - Developer-written descriptions are less so. This is reasoning, not a sourced measurement.
- steam_deck: the Steam Deck launched in 2022 (general knowledge, not sourced here). A verification flag cannot be legitimate information for reviews written before it existed, and it is a scrape-time snapshot for later reviews. The project's own field-selection notes already exclude it on this basis (local file `field-selection.md`, line 37).
- Current price/discount reflect sale state at scrape time, not at review time. Excluding them, or replacing them with time-invariant attributes such as a free-to-play flag where verifiable, follows Kaufman's rule.
- Cold-start claims based on content features should be limited to what the snapshot can support. Content features of items first released after T are legitimate at test time only if the metadata existed at release. That cannot be verified from a single snapshot.

### Gaps
- No peer-reviewed source found that measures how much snapshot item metadata inflates temporal-split recsys results.
- Official Kaggle dataset documentation (collection date, how positive_ratio, user_reviews and the users table were computed) could not be retrieved. These details should be checked directly on the Kaggle page before being cited in a report.
- Steam's rule for the review-summary label (rating) was not verified from Valve documentation.
