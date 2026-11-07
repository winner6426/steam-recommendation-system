# Item content features for Hybrid CF + content and contrastive (InfoNCE) game recommendation

Scope: tags, description, title, price, discount, platform flags (win/mac/linux), Steam Deck, release date. Target model: g_i = V[i] + W·c_i, with c_i from TF-IDF+SVD or a frozen sentence encoder over description+tags, plus InfoNCE with content-based semi-hard negatives.
Research budget was small (about 15 tool calls). Several PDFs could not be parsed, so some findings come only from search snippets or abstracts and are flagged **[snippet only]** or **[unverified]**.

## 1. Game recommendation papers (Steam and others) and hybrid/cold-start models: which content features helped?

### Takeaway
Steam-specific academic evidence is thin and mostly small-scale. The strongest general evidence is that content-to-CF hybrids (LightFM, DropoutNet, Heater, CB2CF, CLCRec) mainly help cold or sparse items and roughly match pure MF on warm items. Text-only item encoders (UniSRec, Recformer) help most in low-resource or cold-start settings, and ID+text is generally best. No source isolates the value of price, platform flags or Steam Deck status for games.

### Cited Findings
- GameRecs (Stefanidis group, Tampere) is a content-based Steam recommender that uses Steam's user-voted tags as game descriptors. It scores games as similar when they share heavily voted tags and removes tags that describe play mode rather than game type. **[snippet only; PDF not parsed]** — [GameRecs PDF](https://homepages.tuni.fi/konstantinos.stefanidis/docs/qauca19.pdf)
- "A Hybrid Recommender System for Steam Games" (Gong, Ye, Stefanidis, ISIP 2019, published 2020) states that it combines playing time, game price, game release date, genres and friends' preferences. The method and results were not verified. **[abstract snippet only]** — [Unpaywall/Springer DOI 10.1007/978-3-030-44900-1_9](https://unpaywall.org/10.1007%2F978-3-030-44900-1_9)
- Bunga, Batista & Ribeiro (2021) built a pure CF Steam recommender that infers explicit ratings from implicit data. It uses no content features. **[snippet only]** — [ISCTE repository](https://repositorio.iscte-iul.pt/bitstream/10071/23677/1/conferenceobject_82722.pdf)
- LightFM (Kula 2015) represents users and items as sums of their metadata-feature latent factors. In cold-start or sparse settings it outperforms both collaborative and content-based models. With abundant interactions it "performs at least as well as" pure MF. Learned feature embeddings encode semantic information, similar to word embeddings. — [arXiv 1507.08439](https://arxiv.org/abs/1507.08439)
- DropoutNet (Volkovs et al., NeurIPS 2017) applies dropout to the preference inputs during training so the network learns to rely on content alone when preference data is missing. It reports state-of-the-art results on public benchmarks, including CiteULike. Warm and cold models are trained separately because their validation sets differ. — [NeurIPS paper](https://proceedings.neurips.cc/paper/2017/file/dbd22ba3bd0df8f385bdac3e9f8be207-Paper.pdf); [code](https://www.github.com/layer6ai-labs/DropoutNet)
- A 2025 follow-up reports that DropoutNet's performance declines sharply when user content is unavailable. — [arXiv 2508.14905](https://arxiv.org/pdf/2508.14905v1)
- Heater (Zhu et al., SIGIR 2020) jointly trains CF and content-transformation functions. It adds a similarity constraint between the transformed content representation and the CF representation, randomized training that feeds either the pretrained CF vector or the transformed content vector, and a mixture-of-experts transformation. It was evaluated on CiteULike, LastFM and XING, and removing any of the three components hurts performance. — [webis anthology](https://ir.webis.de/anthology/2020.sigirconf_conference-2020.119/); [slides](https://zziwei.github.io/pubs/Ziwei_SIGIR_2020_Cold_slides.pdf); [dblp](https://dblp.org/rec/conf/sigir/ZhuSSC20.html)
- CB2CF (Barkan et al., RecSys 2019) learns a mapping from an item's content (text, tags and similar fields, using a CNN for text) to its CF vector from an external CF model. It was designed for Microsoft Store and beats an alternative content-based model on completely cold items in movie and app recommendation. — [arXiv 1611.00384](https://arxiv.org/pdf/1611.00384); [dblp](https://dblp.org/rec/conf/recsys/BarkanKYK19.html)
- CLCRec (Wei et al., ACM MM 2021) treats cold-start as mutual-information maximization. Its loss combines a user–item contrastive term with an item collaborative-embedding vs. content-representation contrastive term, and the CF encoder can be MF or LightGCN. It reports gains in both warm and cold settings on four datasets. — [arXiv 2107.05315](https://arxiv.org/abs/2107.05315); [ar5iv](https://ar5iv.labs.arxiv.org/html/2107.05315)
- UniSRec (Hou et al., KDD 2022) uses item description text encoded by BERT, with parametric whitening and a mixture-of-experts adaptor because raw BERT embeddings are anisotropic and not directly suited to recommendation. The text-only (inductive) and text+ID (transductive) variants both improve, and text+ID is generally best. **[numbers not checked]** — [arXiv 2206.05941](https://arxiv.org/pdf/2206.05941)
- Recformer (Li et al., KDD 2023) flattens item key–value attributes such as title and brand into a text "sentence". Its largest gains are in low-resource and cold-start settings across six datasets. — [arXiv 2305.13731](https://arxiv.org/pdf/2305.13731)
- The SASRec Steam benchmark (Kang & McAuley) is ID-only. Interactions are stored as user id / item id sorted by timestamp, so it says nothing about content features. — [SASRec repo](https://www.github.com/kang205/SASRec)

### Inferences
- In the planned g_i = V[i] + W·c_i, the content term should mainly be judged on cold and low-interaction item buckets. On warm items the literature (LightFM) predicts parity with MF, not large gains, so report results stratified by item popularity and train-time interaction count.
- CLCRec's item-content contrastive term and Heater's similarity constraint both show that aligning V[i] with W·c_i is a well-studied extra loss. It is optional for Phase 2 but would change the Phase-2/Phase-3 comparison if added to only one phase.
- An additive V[i] lets warm items ignore content. DropoutNet-style dropout of V[i] (or Heater's randomized training) is the standard way to force W·c_i to be informative for cold items. This would be a design choice to document, not a default.

### Gaps
- I found no Steam paper with an ablation that isolates tags, description, price, platform flags or release date. Gong et al. (2020) may contain one but could not be read.
- I found no academic work on Steam Deck compatibility or platform flags (win/mac/linux) as recommendation features. Their usefulness is **unverified**. A reasonable prior is that they are low-information for Windows-dominant users and partly redundant with release era. That is an inference, not a finding.
- Title as a stand-alone feature has no direct evidence. Recformer includes the title inside its attribute text.

## 2. Tags vs. description, and TF-IDF vs. Sentence-BERT

### Takeaway
There is no rigorous head-to-head of Steam tags vs. descriptions. The one Steam-specific comparison found (an Estonian bachelor's thesis) reports that a TF-IDF hybrid beat an SBERT hybrid [snippet only]. RecSys 2024 evidence (beeFormer) shows that off-the-shelf sentence encoders capture semantic similarity rather than interaction similarity. They are only roughly on par with simple CF baselines in time-split and are clearly beaten by encoders fine-tuned on interactions. Neither TF-IDF nor a frozen SBERT should be assumed to win, so test both.

### Cited Findings
- A University of Tartu bachelor's thesis compares a TF-IDF content-similarity hybrid and an SBERT-embedding hybrid on Steam data, both combined with overall ratings. It reports that the TF-IDF version outperformed the SBERT version. **[snippet only; PDF not parseable; metrics and setup unverified]** — [UT DSpace](https://dspace.ut.ee/bitstreams/a78db95c-0396-4f7c-9e1c-671e8d619ce8/download)
- GameRecs relies on user-voted Steam tags weighted by vote count and drops play-mode tags such as single-player/multiplayer. This suggests tag vote weight and tag-type filtering matter. **[snippet only]** — [GameRecs PDF](https://homepages.tuni.fi/konstantinos.stefanidis/docs/qauca19.pdf)
- beeFormer (Vančura, Kordík, Straka, RecSys 2024) argues that sentence Transformers trained for semantic similarity miss interaction patterns. Its datasets are ML20M, Goodbooks-10k and Amazon Books (no games), with LLM-standardized descriptions.
  - Item cold-start with the Heater mapping, Recall@20: off-the-shelf all-mpnet 0.2078, nomic 0.2154 and bge-m3 0.2052 on GB10k, vs. 0.2710 for an interaction-fine-tuned encoder. On ML20M the off-the-shelf models score 0.3114 / 0.3049 / 0.2847 vs. 0.3204.
  - Amazon Books time-split, R@20: zero-shot all-mpnet 0.0218, nomic 0.0387 and bge-m3 0.0398, vs. KNN 0.0370, ALS 0.0344, ELSA 0.0367 and SANSA 0.0421. beeFormer reaches 0.0649 zero-shot and 0.0706 supervised.
  — [arXiv 2409.10309](https://arxiv.org/html/2409.10309v2)
- The choice of encoder matters: all-mpnet trailed all CF baselines in time-split, while nomic and bge-m3 were roughly on par. — [arXiv 2409.10309](https://arxiv.org/html/2409.10309v2)
- UniSRec found raw BERT text embeddings anisotropic and added parametric whitening and an MoE adaptor before using them. — [arXiv 2206.05941](https://arxiv.org/pdf/2206.05941)
- General commentary: TF-IDF matches vocabulary rather than meaning, and pre-trained sentence encoders give embeddings usable immediately for new items but do not model personal preference. **[secondary/non-peer-reviewed sources]** — [Manning ch.3 preview](https://www.manning.com/preview/ai-recommender-systems/chapter-3); [DEV article](https://dev.to/nithi_sara/where-nlp-meets-recommendations-1g9j)
- A 2025 paper notes that BERT is resource-intensive compared with TF-IDF. — [arXiv 2502.15223](https://arxiv.org/pdf/2502.15223)

### Inferences
- Steam tags are a short, curated, community-voted vocabulary close to "genre/mechanic". Descriptions are long, marketing-heavy and include boilerplate such as DLC and edition text. For TF-IDF+SVD, tags alone or tag-weighted text may be less noisy. For a frozen encoder, concatenating tags into the text likely helps, but this is untested here. Recommended ablation: tags-only, description-only, tags+description, for each encoder.
- Because W is trainable in g_i = V[i] + W·c_i, the "semantic vs. interaction similarity" mismatch that beeFormer identifies is partly corrected by W. A frozen encoder can still be adequate, which matches beeFormer's Heater-mapping results being competitive.
- Consider whitening or normalizing c_i (UniSRec) before W, especially for SBERT vectors.
- Title text has high overlap across franchise and edition items (sequels, "Deluxe Edition"). It may create near-duplicate content vectors, which matters for content-based negatives (Section 5).

### Gaps
- No verified Steam head-to-head of tags vs. description with Recall/NDCG numbers was found. The York and Politecnico di Torino theses ([York](https://etheses.whiterose.ac.uk/id/eprint/36587/1/Oyebiyi_207062611_CorrectedThesisClean.pdf), [PoliTo](https://webthesis.biblio.polito.it/22582/1/tesi.pdf)) may compare TF-IDF to BERT for cold-start, but they were not read.
- The Tartu thesis result needs verification from the full PDF, which was downloaded but could not be text-extracted in this environment.

## 3. Price and discount as features: snapshot vs. price at interaction time

### Takeaway
I found no recommender paper that directly studies "catalog price snapshot vs. price at interaction time". The general temporal-leakage literature and Steam pricing dynamics both imply that a scraped current price or discount is a post-hoc snapshot. Steam prices fall substantially over a game's life and spike down during sales, so using the snapshot as a feature for past interactions is anachronistic and possibly leaky. Discount in particular is a transient state at scrape time.

### Cited Findings
- Offline evaluations that ignore the global timeline cause data leakage, and access to different amounts of future data can raise or lower measured accuracy. — [Ji et al., arXiv 2010.11060](https://arxiv.org/abs/2010.11060v2)
- Time-aware evaluation should make each prediction from only the information known at prediction time. — [arXiv 2210.04149](https://ar5iv.labs.arxiv.org/html/2210.04149)
- Price is used as a personalization feature in e-commerce, for example as the ratio of the current item's price to the average price of previously clicked items. — [CEUR ComplexRec paper](https://ceur-ws.org/Vol-2697/paper4_complexrec.pdf)
- Steam price dynamics (PC Gamer analysis of price data): the median time to reach 80–90% discount tiers was 446 days (range 261–611). Less commercially successful $30+ games dropped in price about 60% faster than bestsellers. This makes price correlated with popularity and age. **[journalism, not peer-reviewed]** — [PC Gamer](https://www.pcgamer.com/how-long-does-it-take-games-to-be-cheap-on-steam/3)
- A Tilburg study reports that a high initial discount and mixed reviews lead to faster discounting (two genres only). This links current price/discount to review sentiment, which overlaps the review-derived features the project bans. **[thesis, limited scope]** — [Tilburg](https://arno.uvt.nl/show.cgi?fid=161709)
- A 2014 NPD survey found that about 50% of US PC gamers wait for discounts before buying. — [PC Gamer/NPD](https://www.pcgamer.com/pc-owners-more-likely-to-wait-for-price-drops-before-purchasing-according-to-npd/)
- SteamSpy's author says sales increase purchases. One developer reported a 60x sales increase at 75% off. **[anecdotal]** — [SteamSpy Medium](https://medium.com/steam-spy/about-steam-summer-sale-bab1acc3e24); [Game Developer](https://gamedeveloper.com/disciplines/how-players-shop-during-a-steam-sale)

### Inferences
- A snapshot `price_final` / `discount` reflects the scrape date, not review time. Because price decays faster for less successful games, the snapshot carries future-outcome information. That is a mild form of the leakage AGENTS.md forbids for `positive_ratio` and similar fields. Recommendation: exclude `discount` entirely. If price is used, prefer `price_original` (list price, more stable) in coarse log-bins, and document it as a snapshot limitation.
- The price-at-interaction effect (buying during sales) cannot be modeled without historical price data. Claims about price sensitivity should be avoided.
- Free-to-play (price = 0) is a qualitatively distinct, stable category and is likely the most defensible price-derived signal. This is an inference.

### Gaps
- No peer-reviewed RecSys study quantifying the leakage or bias from snapshot price features was found.
- No dataset of historical Steam prices tied to review timestamps was identified in this pass.

## 4. Release date / item age as feature and as candidate filter

### Takeaway
Item age strongly shapes popularity and selection bias, which shift with time since release. Release date is a legitimate pre-interaction attribute: it is known at release and does not leak, unlike review aggregates. As a candidate filter, "released ≤ cutoff" is required for temporal correctness. Age-as-feature mostly encodes a popularity/recency prior, so evaluate it separately from content effects.

### Cited Findings
- On MovieLens, item popularity changes with item age (time since publication), so selection bias is dynamic. DANCER extends IPS to handle age-dependent bias and improves over static-bias debiasing. — [Huang, Oosterhuis, de Rijke, WSDM 2022, arXiv 2111.12481](https://arxiv.org/abs/2111.12481)
- Items need time to gain popularity, so recommended items tend to be old. Boosting by recent degree growth significantly improves accuracy in network-based recommendation. — [Vidmer & Medo, arXiv 1704.08027](https://arxiv.org/pdf/1704.08027)
- Freshness is often used as a discriminative factor between candidate items in time-aware recommenders, and decayed popularity weights older interactions less. — [arXiv 1807.04204](https://arxiv.org/pdf/1807.04204.pdf); [SciTePress 2025](https://scitepress.org/publishedPapers/2025/142869/pdf/index.html)
- Gong et al. (2020) use release date as an input in a Steam hybrid. **[abstract only]** — [DOI 10.1007/978-3-030-44900-1_9](https://unpaywall.org/10.1007%2F978-3-030-44900-1_9)
- Timeline-ignoring evaluation leaks future information. Recommending items not yet released at the cutoff is one instance. — [arXiv 2010.11060](https://arxiv.org/abs/2010.11060v2)

### Inferences
- Candidate filter: exclude games whose release date (or first review date, if the release date is missing or unreliable) is after the cutoff T1/T2. Items first appearing between T1 and T2 are true cold-start test items. These are the only place where the content term can be fairly credited with cold-start gains.
- Age as a feature: encode age relative to the cutoff (cutoff − release date, bucketed). Absolute year drifts between T1 and T2 snapshots. Since age correlates with popularity, report a popularity baseline and a "content without age" ablation.
- Pitfall: store release dates can be re-dated (early access → full release) or missing. Verify against the earliest review timestamp in the data. This is an inference, untested here.

### Gaps
- No game-specific study measuring release date as a feature on Recall/NDCG was found.

## 5. Content-based hard / semi-hard negatives and false-negative risk

### Takeaway
Hard negatives are informative but disproportionately likely to be false negatives. In implicit CF, many "unknown" items that score high are items the user would like. SRNS (NeurIPS 2020) uses the observation that false negatives have stably high scores over training. Robinson et al. (ICLR 2021) show that pushing hardness up (large β) hurts when positives/negatives are only approximately known. Content-similar negatives (near-duplicate tags or description) are especially risky for games because of sequels, editions and genre clusters. Use them as an ablation with controllable hardness and false-negative safeguards, not as the main sampler. This matches AGENTS.md.

### Cited Findings
- SRNS (Ding, Quan, Yao, Li, Jin, NeurIPS 2020): only a few negatives are important for learning, and false negatives tend to have stable (low-variance) high predictions across training iterations. SRNS keeps a small memory of candidate negatives and favors those with both high score and high variance. It reports 0.77–8.4% gains over prior samplers on two synthetic and three real datasets. — [arXiv 2009.03376](https://arxiv.org/abs/2009.03376v1); [NeurIPS abstract](https://papers.nips.cc/paper/2020/hash/0c7119e3a6a2209da6a5b90e5b5b75bd-Abstract.html); [PDF](https://fi.ee.tsinghua.edu.cn/~dingjingtao/papers/SRNS-NeurIPS20.pdf); [reviews](https://papers.nips.cc/paper/2020/file/0c7119e3a6a2209da6a5b90e5b5b75bd-Review.html)
- Robinson, Chuang, Sra, Jegelka (ICLR 2021), "Contrastive Learning with Hard Negative Samples": a tunable concentration β upweights negatives similar to the anchor, with positive-unlabeled debiasing via the class prior τ⁺, implemented as importance reweighting.
  - Larger β targets the negatives most likely to be false negatives.
  - In vision, larger β did not reliably help because the false-negative correction is only approximate. With true labels, performance rose monotonically with β.
  - Annealing β toward 0 reduced sensitivity to the initial β.
  - τ⁺ > 0 caused numerical issues for sentence embeddings.
  - Gains were slight or vanished at larger batch/negative counts N on CIFAR10.
  — [arXiv 2010.04592](https://arxiv.org/abs/2010.04592); [ar5iv](https://ar5iv.labs.arxiv.org/html/2010.04592)
- CLCRec treats unobserved user–item pairs as negatives in its contrastive U–I term, so it inherits the standard false-negative issue. — [arXiv 2107.05315](https://arxiv.org/abs/2107.05315)

### Inferences
- In this dataset, "unknown" (not reviewed) does not mean disliked, and `users.csv` lacks ownership. Content-nearest unknown games to a user's positives are, by construction, the items a content-based recommender would recommend, so their false-negative rate is likely elevated. Safeguards:
  1. Sample semi-hard negatives from a similarity band (for example, rank percentile 5–30% rather than top-k nearest).
  2. Exclude same-franchise or near-duplicate titles.
  3. Mix with random negatives at a fixed ratio.
  4. Optionally use SRNS-style score-variance filtering.
  5. Keep explicit `is_recommended=False` as the only "sure" hard negatives.
- If content vectors c_i are both the model input and the negative-mining similarity, the model is trained to separate items its own content tower deems similar. This can conflict with W·c_i, and it should be noted as a design tension. A different representation for mining (e.g., tags-only Jaccard) vs. input (text encoder) is one mitigation. This is an inference, untested.
- Following Robinson et al., report sensitivity to the hardness knob (band or β) and to the number of negatives. The benefit may disappear with large in-batch negative counts.

### Gaps
- No paper was found that evaluates content-based (metadata-similarity) hard negatives specifically for game or Steam recommendation, or that measures their false-negative rate.
- The rest of the Robinson et al. paper (beyond the first 100k characters) and SRNS's per-dataset tables were not read.
