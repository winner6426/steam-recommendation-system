# Steam datasets in recommender-system research: which fields papers use

Scope: (a) the Kaggle "Game Recommendations on Steam" dataset (Anton Kozyriev); (b) the UCSD McAuley-lab Steam datasets (Pathak et al. SIGIR 2017; Kang & McAuley ICDM 2018; Wan & McAuley RecSys 2018). Research date: 2026-11-18. Tool budget was about 20 calls, so the literature search for (a) is not exhaustive.

## Q1. What does the Kaggle dataset documentation say about `hours`, helpful/funny, positive_ratio, user_reviews, price, steam_deck and snapshot date?

### Takeaway
The official Kaggle dataset card does not say whether `hours` is playtime at review time or at scrape time. Through the public API, the card has **no per-column descriptions** and **no snapshot/crawl date** for any field. The only dates are version metadata: the last update (version 28) was on 2024-08-14, and the expected update frequency is "never". The time semantics of `hours`, helpful/funny, positive_ratio, user_reviews, price and steam_deck are therefore undocumented. The safest assumption is that each is a single crawl-time snapshot.

### Cited Findings
- Full official description (verbatim, from the Kaggle API metadata): "The dataset contains **over 41 million** cleaned and preprocessed user recommendations (reviews) from a Steam Store … Additionally, it contains detailed information about games and add-ons." — [Kaggle API dataset view](https://www.kaggle.com/api/v1/datasets/view/antonkozyriev/game-recommendations-on-steam); [Kaggle page](https://www.kaggle.com/datasets/antonkozyriev/game-recommendations-on-steam)
- Verbatim file descriptions in the card:
  - games.csv: "a table of games (or add-ons) information on ratings, pricing in US dollars $, release date, etc. A piece of extra non-tabular details on games, such as descriptions and tags, is in a metadata file".
  - users.csv: "a table of user profiles' public information: the number of purchased products and reviews published".
  - recommendations.csv: "a table of user reviews: whether the user recommends a product. The table represents a many-many relation between a game entity and a user entity."
  - Source: [Kaggle API metadata](https://www.kaggle.com/api/v1/datasets/metadata/antonkozyriev/game-recommendations-on-steam)
- The card states: "A preprocessing pipeline anonymized all user IDs. All collected data is accessible to a member of the general public." The source is given as "Steam Official Store (https://store.steampowered.com)". — [Kaggle API metadata](https://www.kaggle.com/api/v1/datasets/metadata/antonkozyriev/game-recommendations-on-steam)
- Metadata facts:
  - License: CC0-1.0.
  - `expectedUpdateFrequency`: "never".
  - Current version 28, "Update 2024-08-14" (lastUpdated 2024-08-14T13:15:03Z).
  - File creation dates 2024-08-14. Sizes: recommendations.csv about 2.02 GB, users.csv about 193 MB, games_metadata.json about 17.5 MB, games.csv about 4.9 MB.
  - The file-list API returns `"columns":[]` and an empty `description` for every file, so the API exposes no column-level documentation.
  - Sources: [Kaggle API view](https://www.kaggle.com/api/v1/datasets/view/antonkozyriev/game-recommendations-on-steam); [Kaggle API file list](https://www.kaggle.com/api/v1/datasets/list/antonkozyriev/game-recommendations-on-steam)
- DataCite DOI: 10.34740/KAGGLE/DS/2871694, creator Anton Kozyriev, publication year 2023, registered 2023-02-07. — [DataCite](https://api.datacite.org/dois/10.34740/kaggle/ds/2871694); [DataCite Commons](https://commons.datacite.org/doi.org/10.34740/kaggle/ds/2871694)
- A third-party GitHub project gives column glosses: "is_recommended: Whether the user recommends the game or not; hours: Hours of playtime the user has on the game". This may have been copied from the Kaggle web UI's column panel. It does not settle review-time vs scrape-time. — reported via search snippet of [jachian/Steam-Recommender-System](https://github.com/jachian/Steam-Recommender-System) (UNVERIFIED: I did not open the repo; the snippet attribution comes from the search tool)
- Kaggle's HTML page is rendered with JavaScript and returned only its title to WebFetch, so I could not inspect any column descriptions shown in the web UI. — [Kaggle page](https://www.kaggle.com/datasets/antonkozyriev/game-recommendations-on-steam)

### Inferences
- Because the card gives one crawl history (first version Feb 2023, last 2024-08-14) and no per-field timestamps, `hours`, `helpful`, `funny`, and all games.csv aggregates (`rating`, `positive_ratio`, `user_reviews`, prices, `discount`, `steam_deck`) should be treated as values at crawl time of the version in use. They are not values at review `date`.
- Steam's own review UI shows "hrs on record" at the time a page is viewed, plus a separate "at review time" value. This is general knowledge and was not verified in this session. If the crawler took the on-record figure, `hours` is at scrape time and leaks future information relative to the review `date`.
- This supports the project rule (AGENTS.md) of avoiding positive_ratio/rating/user_reviews/helpful/funny as features. `hours` carries the same leakage risk and should be treated as a snapshot. One empirical check: look for reviews whose `hours` greatly exceeds the playtime possible between release and review date.
- `discount` and `price_final` reflect a single crawl moment, possibly during a sale, so they are noisy as content features.

### Gaps
- No official statement on `hours` semantics (review time vs scrape time) was found anywhere. The author published no paper or GitHub crawler that I could find.
- I could not verify the exact column-description text in Kaggle's web UI (it needs JavaScript). Opening the page in a browser, or checking the Kaggle "Data Card" column panel, would settle this.
- No snapshot date for `steam_deck`, prices or `positive_ratio` beyond the version date 2024-08-14.

## Q2. Which papers (2018–2026, incl. arXiv) use the Kozyriev dataset, and which fields?

### Takeaway
Peer-reviewed use of the Kozyriev dataset is rare in what I could find. One clear arXiv paper (FairGap, arXiv 2608.08284, 2026) uses it as an LLM-recommender fairness benchmark and does not name the fields it uses. Most other uses are GitHub/course projects. No paper I found documents a temporal split with `hours` or `is_recommended` semantics in detail.

### Cited Findings
- **"Fair on the Surface? Benchmarking Hidden–Output Fairness Gaps in LLM Recommenders" (arXiv 2608.08284v1, 2026)** uses the dataset as its game domain ("SteamReviews"):
  - It has 136,101 users, 7,762 items, average history 14.49 items, median 13.
  - Users need ≥10 valid interactions, and profiles are truncated to ≤20 items "ordered by rating priority and then recency".
  - It does not name the columns used (no explicit mention of is_recommended, hours, tags), it gives no positive definition, and it has no Steam-specific split.
  - It cites the dataset as "A. Kozyriev … 2021, Game recommendations on steam. Kaggle dataset". That year conflicts with DataCite's 2023.
  - Sources: [arXiv HTML](https://arxiv.org/html/2608.08284v1); [Pith summary](https://pith.science/paper/2608.08284). I read only the first 100k of 124k characters.
- **sunaggn/steam-nffmtl (GitHub)** is a "leakage-controlled re-implementation of the NFF-MTL experiments".
  - It uses recommendations.csv and games_metadata.json, with tag fusion.
  - It has a main recommendation head plus an auxiliary playtime head, so `hours` serves as a multi-task regression target.
  - Relevance is configurable: positive-review vs any-review sensitivity.
  - Its protocol includes a history cutoff, a test year, and validation users.
  - Its lineage is melisguclu/game-recommender-system. The original NFF-MTL paper's authors and venue are not given.
  - Source: [GitHub](https://github.com/sunaggn/steam-nffmtl)
- **juanthom99/steam-game-recommender (course project)** uses the Kozyriev data ("41M reviews with hours") only for "extra training rows, label validation". McAuley Steam data serves as its main train/test data. — [GitHub](https://github.com/juanthom99/steam-game-recommender) (via search snippet)
- Other GitHub projects using it:
  - [raghazzz/steam-game-recommender](https://github.com/raghazzz/steam-game-recommender): after cleaning, "2.15 million likes from 197,166 players on 10,557 games", which implies is_recommended=True is used as positive (per search snippet, not opened).
  - A project that samples 50,000 users, giving 22,521 games and 642,760 interactions (search snippet, project unclear).
  - Others: [cnathan/steam-game-recommender](https://github.com/cnathan/steam-game-recommender), [fccacioli/steam-game-recommendations-neo4j](https://github.com/fccacioli/steam-game-recommendations-neo4j).
- **CPGRec, "Category-based and Popularity-guided Video Game Recommendation" (Li, Ma, Liu, Feng, Zhang, Wang; WWW 2024; arXiv 2604.14598)** uses "the Steam dataset" with category and popularity signals and negative-sample reweighting. Neither the abstract nor the repo README identifies the source dataset. — [arXiv](https://arxiv.org/abs/2604.14598); [GitHub](https://github.com/CPGRec2024/CPGRec). UNVERIFIED which Steam dataset it uses; it is probably not Kozyriev.
- Papers that look relevant but use other data:
  - "Steam Recommendation System" (arXiv 2305.04890, Spark ALS) uses McAuley's Steam data, not Kozyriev. — [arXiv PDF](https://arxiv.org/pdf/2305.04890)
  - arXiv 2202.03392 ("Large-scale Personalized Video Game Recommendation…") also appeared in search; its dataset was not checked. — [arXiv](https://arxiv.org/pdf/2202.03392)

### Inferences
- The Kozyriev dataset is mainly a hobby/teaching dataset. Little peer-reviewed literature relies on its specific fields, so a project that defines a careful temporal protocol (positive = is_recommended True, explicit negatives = False, `hours` treated as a snapshot) has no strong precedent and should justify its choices itself.
- Where `hours` is used at all (e.g., NFF-MTL), it is an auxiliary target, not an input weight.

### Gaps
- Google Scholar and Semantic Scholar citation searches on DOI 10.34740/KAGGLE/DS/2871694 were not done. DataCite reports 0 citations, but Kaggle DOIs are poorly tracked. More papers, especially IEEE/Springer conference papers, probably exist but were not found.
- I could not identify the original NFF-MTL paper.
- CPGRec's dataset source is unverified; its `steam_data` folder would need inspecting.

## Q3. In the UCSD Steam data, how were playtime, the recommend flag, tags, genres, price, release date and developer used?

### Takeaway
All three canonical UCSD papers use only interaction signals: purchase (ownership), play (playtime > 0), review presence, and the recommend flag. None of them uses tags, genres, price, release date or developer as model features. SASRec treats any review as implicit positive, regardless of recommend/not-recommend, orders reviews by timestamp, and uses leave-last-out with 100 sampled negatives. chainRec turns playtime and the recommend flag into binary stages of a monotonic chain. Pathak et al. use purchases with BPR and popularity-matched negative sampling and a random 70/10/20 split.

### Cited Findings
**Dataset page (UCSD)**
- The Steam dataset page describes reviews plus bundle info. Stats: 7,793,069 reviews, 2,567,538 users, 15,474 items, 615 bundles.
- Its metadata lists "reviews, purchases, plays, recommends ('likes'), product bundles, pricing information". Tags are not listed.
- Files: australian_user_reviews.json.gz and australian_users_items.json.gz (v1), steam_reviews.json.gz and steam_games.json.gz (v2), bundle_data.json.gz.
- The page asks users to cite all three papers below.
- Source: [UCSD datasets page](https://cseweb.ucsd.edu/~jmcauley/datasets.html)

**Pathak, Gupta & McAuley, SIGIR 2017 (bundles)**
- Data were crawled from the Australian "GameAus" community: 88,310 users, 10,978 games, 902,967 game purchases, 615 bundles, 87,565 bundle purchases.
- The data are described as including "what games users purchased, whether they played them or not after purchase, and how they rated them".
- Interactions: item BPR on purchases. A positive is a purchased item; a negative is an unpurchased item, drawn with a graph-sampling method so negatives match the positives' degree (popularity) distribution. This counters the popularity bias that uniform sampling caused.
- Bundle-model features: item BPR latent factors, bundle size, and bundle correlation (mean pairwise Pearson correlation of item latent factors).
- Split: a random 70/10/20 train/val/test split of the sampled triplets. Metric: AUC.
- Pricing/discount is explicitly left to future work ("investigate the role that pricing (and in particular, discount rate) has"), so price was not used.
- Source: [SIGIR'17 PDF](https://cseweb.ucsd.edu/~jmcauley/pdfs/sigir17.pdf)

**Kang & McAuley, SASRec, ICDM 2018**
- Steam is described as "a new dataset crawled from Steam … 2,567,538 users, 15,474 games and 7,793,069 English reviews spanning October 2010 to January 2018." It "also includes rich information that might be useful in future work, like users' play hours, pricing information, media score, category, developer (etc.)". So these fields were not used.
- Preprocessing: "we treat the presence of a review or rating as implicit feedback … and use timestamps to determine the sequence order"; users and items with fewer than 5 actions are discarded. Not-recommended reviews therefore count as positives.
- Split: per user, the most recent action is test, the second most recent is validation, and the rest is training. This is temporal per user (leave-last-out), not a global cutoff.
- Training uses 1 random negative per time step. Evaluation uses 100 randomly sampled negatives plus the ground truth (Hit@10, NDCG@10).
- Source: [arXiv 1808.09781 PDF](https://arxiv.org/pdf/1808.09781)

**Wan & McAuley, chainRec, RecSys 2018 (DOI 10.1145/3240323.3240369)**
- Uses the Australian Steam data from [21] (Pathak et al.), including "users' purchase information, play time of games they purchased, reviews, and thumbs-ups (i.e., 'recommended' or 'not recommended')". From these it builds a "purchase → play → review → recommend" chain per user-item pair.
- Playtime is binarized as play vs never played: "around 36% of purchased games are never played".
- Steam stats after preprocessing: 2,447,847 interactions, with stage rates purchase 100%, play 64.0%, review 2.2%, recommend 2.0%.
- Not-recommended reviews reach the review stage but not the recommend stage. The paper discusses the difficulty of comparing "reviewed but not recommended" against "purchased but not reviewed".
- Preprocessing: users who never reached the last stage are discarded, as are items with fewer than 5 interactions.
- Split: 100,000 interaction chains sampled for validation and another 100,000 for test, each chain from a different user. The selection is random, not temporal.
- Metrics: AUC and NDCG. N=1 contrastive negative per positive.
- itemPop is a strong baseline on Steam, "possibly because the collected users' reviews are biased towards popular games".
- The authors suggest play counts could be used as confidence weights in future work, so hours were not used as weights. No content features were used for Steam; genre/language features were analysed only for Goodreads.
- Code: github.com/MengtingWan/chainRec.
- Sources: [RecSys'18 PDF](https://cseweb.ucsd.edu/%7Ejmcauley/pdfs/recsys18b.pdf); [slides](https://mengtingwan.github.io/paper/recsys18_mwan_slides.pdf); [dblp](https://dblp.org/rec/conf/recsys/WanM18.html)

### Inferences
- Precedent for this project:
  - SASRec-style work on Steam ignores the recommend flag and treats all reviews as positive.
  - chainRec is the canonical precedent for treating "reviewed but not recommended" as a distinct weaker state, not a hard negative.
  - Neither paper uses playtime as a continuous weight.
- None of the three canonical papers uses game content (tags/genres/developer/price/release date). A Hybrid CF + content model therefore departs from these baselines, and its content features must be justified separately.
- SASRec's leave-last-out with 100 sampled negatives is the protocol AGENTS.md rules out as a main result (sampled ranking). A global temporal cutoff with full-catalog evaluation is stricter than all three precedents.

### Gaps
- The exact field names in the v2 files (e.g., `hours` and `date` in steam_reviews.json.gz, and genres/tags/specs/price/release_date/developer in steam_games.json.gz) were not verified from the files or the page this session. They are known from general knowledge only (UNVERIFIED).
- Whether UCSD's review `hours` was crawled as "hours on record at crawl time" (likely, Jan 2018 crawl) is undocumented.
- I did not check later papers that reuse UCSD Steam with content features (e.g., CIKM/KDD sequential papers using Steam item attributes such as S3-Rec).
