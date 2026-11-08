# Chọn trường Steam theo luật "không cỗ máy thời gian"

Tài liệu recommender chọn trường theo hai câu hỏi: trường đó **đã biết trước thời điểm dự đoán chưa** (luật "no-time-machine" của Kaufman et al.), và trường đó có **trả lời đúng câu hỏi của thí nghiệm** hay không. Áp vào dữ liệu Kozyriev, kết luận như sau. Pha 1 chỉ cần `user_id`, `app_id` và nhãn `is_recommended` (True là positive, False là explicit negative). `date`, `review_id` và `date_release` chỉ dùng để chia `T1`/`T2`, khử trùng lặp và lọc candidate. Pha 2 thêm đúng một nguồn content là `description` + `tags` → `c_i`. Pha 3 không thêm trường nào và chỉ đổi loss. Mọi trường tổng hợp đến lúc thu thập đều bị loại: `positive_ratio`, `rating`, `user_reviews`, `helpful`, `funny`, `users.products`, `users.reviews`. Giá hiện tại, `discount` và `steam_deck` cũng bị loại. `hours` chỉ nên là ablation, vì Kaggle không cho biết đây là giờ chơi lúc viết review hay lúc thu thập. Ba bài kinh điển trên Steam của nhóm UCSD đều chỉ dùng tín hiệu tương tác, không dùng tags, giá hay ngày phát hành. Vì vậy phần content của Pha 2 không có tiền lệ Steam mạnh, và cần được chứng minh bằng kết quả phân tầng theo độ phổ biến của item.

## Bảng kết luận theo từng trường

Chú thích nhãn: **Dùng** là đầu vào chính. **Ablation tùy chọn** là thử nghiệm phụ, có đối chứng, không thuộc cấu hình chính. **Không dùng** là loại khỏi mô hình và khỏi việc chọn hyperparameter. **Chỉ xử lý dữ liệu** là chỉ dùng để split, dedup hoặc lọc candidate, không làm đặc trưng.

| File.trường | Pha 1 | Pha 2 | Pha 3 | Lý do ngắn | Nguồn |
|---|---|---|---|---|---|
| `recommendations.user_id` | Dùng | Dùng | Dùng | Chỉ số hàng của ma trận `U`. Mapping chỉ fit trên dữ liệu đến cutoff | [Rendle 2009](https://arxiv.org/pdf/1205.2618) |
| `recommendations.app_id` | Dùng | Dùng | Dùng | Chỉ số của `V[i]` và khóa nối với content | [Kula 2015](https://arxiv.org/abs/1507.08439) |
| `recommendations.is_recommended` | Dùng | Dùng | Dùng | True là positive. False là explicit negative, có trọng số `w_neg` riêng, tách khỏi unknown | [Hu 2008](http://yifanhu.net/PUB/cf.pdf), [YouTube 2023](https://arxiv.org/abs/2308.12256) |
| `recommendations.date` | Chỉ xử lý dữ liệu | Chỉ xử lý dữ liệu | Chỉ xử lý dữ liệu | Split toàn cục `T1`/`T2`, lịch sử ≤ cutoff, giữ review mới nhất của mỗi cặp | [Ji 2023](https://arxiv.org/abs/2010.11060v4), [Meng 2020](https://arxiv.org/pdf/2007.13237) |
| `recommendations.review_id` | Chỉ xử lý dữ liệu | Chỉ xử lý dữ liệu | Chỉ xử lý dữ liệu | Khóa duy nhất, phá thế hòa khi trùng ngày | (quy ước dự án) |
| `recommendations.hours` | Ablation tùy chọn | Ablation tùy chọn | Ablation tùy chọn | Trọng số tin cậy `1+α·log(1+h)` nhân vào nhãn quan sát được (cả positive lẫn negative). Bắt buộc có đối chứng α=0. Có nguy cơ rò rỉ snapshot và long-game bias | [Hu 2008](http://yifanhu.net/PUB/cf.pdf), [Zhan 2022](https://arxiv.org/abs/2206.06003) |
| `recommendations.helpful` | Không dùng | Không dùng | Không dùng | Vote của người khác, cộng dồn sau khi review đăng. Đo độ hữu ích của review, không đo sở thích. Không có tiền lệ dùng làm trọng số CF | [Kaufman 2011](https://www.cs.umb.edu/~ding/history/470_670_fall_2011/papers/cs670_Tran_PreferredPaper_LeakingInDataMining.pdf) |
| `recommendations.funny` | Không dùng | Không dùng | Không dùng | Như `helpful`. Còn hay rơi vào review đùa hoặc review tiêu cực | như trên |
| `games.app_id` | Chỉ xử lý dữ liệu | Dùng | Dùng | Pha 1 chỉ dùng để nối với `date_release` khi lọc. Pha 2/3 là khóa nối content | — |
| `games.title` | Không dùng | Ablation tùy chọn | Ablation tùy chọn | Gần như định danh. Sequel và edition dễ tạo vector trùng. Không có bằng chứng riêng | [Recformer 2023](https://arxiv.org/pdf/2305.13731) |
| `games.date_release` | Chỉ xử lý dữ liệu | Chỉ xử lý dữ liệu | Chỉ xử lý dữ liệu | Lọc candidate: chỉ giữ game phát hành ≤ cutoff. Tuổi game tương đối với cutoff chỉ nên là ablation riêng, vì nó mã hóa độ phổ biến | [Ji 2023](https://arxiv.org/abs/2010.11060v4), [Huang 2022](https://arxiv.org/abs/2111.12481) |
| `games.win`/`mac`/`linux` | Không dùng | Ablation tùy chọn | Ablation tùy chọn | Tương đối ổn định nhưng ít thông tin. Không tìm thấy bằng chứng nào | (không có nguồn) |
| `games.rating` | Không dùng | Không dùng | Không dùng | Hàm của `positive_ratio` và số review, tức thông tin tương lai | [Sun 2023](https://ar5iv.labs.arxiv.org/html/2210.04149) |
| `games.positive_ratio` | Không dùng | Không dùng | Không dùng | Tổng hợp mọi review, kể cả giai đoạn test. Đóng vai proxy của nhãn | [Kaufman 2011](https://www.cs.umb.edu/~ding/history/470_670_fall_2011/papers/cs670_Tran_PreferredPaper_LeakingInDataMining.pdf) |
| `games.user_reviews` | Không dùng | Không dùng | Không dùng | Là "độ phổ biến tương lai". Nếu cần thì tính lại từ review ≤ cutoff | [Sun 2023](https://ar5iv.labs.arxiv.org/html/2210.04149), [Ji 2020](https://arxiv.org/abs/2005.13829) |
| `games.price_final` | Không dùng | Không dùng | Không dùng | Giá tại lúc thu thập, có thể đang sale | [Kaufman 2011](https://www.cs.umb.edu/~ding/history/470_670_fall_2011/papers/cs670_Tran_PreferredPaper_LeakingInDataMining.pdf) |
| `games.price_original` | Không dùng | Ablation tùy chọn | Ablation tùy chọn | Ổn định hơn `price_final`. Nếu thử thì chỉ dùng dạng bin log hoặc cờ free-to-play, và ghi rõ là snapshot | [PC Gamer](https://www.pcgamer.com/how-long-does-it-take-games-to-be-cheap-on-steam/3) (báo chí) |
| `games.discount` | Không dùng | Không dùng | Không dùng | Trạng thái tạm thời lúc thu thập, có tương quan với mức thành công về sau | [Tilburg](https://arno.uvt.nl/show.cgi?fid=161709) (luận văn) |
| `games.steam_deck` | Không dùng | Không dùng | Không dùng | Nhãn chỉ tồn tại sau khi Steam Deck ra mắt, vô nghĩa với review cũ | (suy luận) |
| `users.user_id` | Không dùng | Không dùng | Không dùng | Chỉ số user lấy từ `recommendations` | — |
| `users.products` | Không dùng | Không dùng | Không dùng | Chỉ là số đếm lúc thu thập, không phải danh sách game sở hữu | [Kaggle card](https://www.kaggle.com/api/v1/datasets/metadata/antonkozyriev/game-recommendations-on-steam) |
| `users.reviews` | Không dùng | Không dùng | Không dùng | Số đếm lúc thu thập, gồm cả review sau cutoff | như trên |
| `games_metadata.app_id` | Không dùng | Dùng | Dùng | Khóa nối content | — |
| `games_metadata.description` | Không dùng | Dùng | Dùng | Thành phần của `c_i`. Lợi ích kỳ vọng chủ yếu ở item lạnh hoặc thưa | [Kula 2015](https://arxiv.org/abs/1507.08439), [UniSRec](https://arxiv.org/pdf/2206.05941) |
| `games_metadata.tags` | Không dùng | Dùng | Dùng | Từ vựng ngắn do cộng đồng gán, gần với genre/mechanic. Pha 3 có thể dùng thêm để đào semi-hard negative trong ablation | [GameRecs](https://homepages.tuni.fi/konstantinos.stefanidis/docs/qauca19.pdf) [chỉ snippet] |

Có hai điểm lệch so với `field-selection.md` hiện tại. Thứ nhất, bảng trên cho phép thử `price_original` ở dạng thô như một ablation, trong khi file đó loại mọi trường `price_*`. Đây là suy luận từ ghi chú, không có paper hỗ trợ. Giữ cách loại hẳn cũng hợp lý. Thứ hai, `title` và `win`/`mac`/`linux` ở Pha 3 phải lặp lại đúng lựa chọn của Pha 2, để Pha 3 chỉ khác Pha 2 ở loss.

## Tiền lệ Steam: chỉ dùng ID, và gần như không ai dùng dữ liệu Kozyriev cẩn thận

Ba bài UCSD trên dữ liệu Steam chỉ dùng tương tác. **SASRec coi mọi review là positive ngầm**, không phân biệt recommend hay không. Bài này dùng leave-last-out với 100 negative lấy mẫu, và ghi rằng play hours, giá và developer là "có thể hữu ích trong tương lai" ([Kang & McAuley 2018](https://arxiv.org/pdf/1808.09781)). **chainRec mã hóa purchase → play → review → recommend thành một chuỗi nhị phân**. Ở đó "review nhưng không recommend" là một trạng thái yếu riêng, và giờ chơi chỉ được nhị phân hóa thành chơi hay chưa chơi ([Wan & McAuley 2018](https://cseweb.ucsd.edu/%7Ejmcauley/pdfs/recsys18b.pdf)). Pathak et al. dùng BPR trên purchase với negative khớp độ phổ biến, split ngẫu nhiên 70/10/20, và để giá cho công việc tương lai ([SIGIR 2017](https://cseweb.ucsd.edu/~jmcauley/pdfs/sigir17.pdf)). Với chính bộ Kozyriev, công trình rõ ràng duy nhất tìm được là FairGap. Bài này dùng dữ liệu làm benchmark công bằng cho LLM, nhưng **không nêu cột nào được dùng** và không có split thời gian ([arXiv 2608.08284](https://arxiv.org/html/2608.08284v1)). Repo steam-nffmtl dùng `hours` làm **mục tiêu phụ (multi-task)** chứ không làm trọng số đầu vào ([GitHub](https://github.com/sunaggn/steam-nffmtl)). Hệ quả là giao thức hai cutoff, chấm full-catalog của dự án chặt hơn mọi tiền lệ. Lựa chọn trường của dự án phải tự biện minh, không thể dựa vào "người khác đã làm vậy".

## Nhãn False nên là nhãn huấn luyện, còn hours chỉ là ablation

Hu et al. nhị phân hóa preference và dùng độ tin cậy `c = 1 + α·log(1 + r/ε)`. Họ thừa nhận việc **gán cùng một mức tin cậy thấp cho mọi cặp bằng 0** là một hạn chế ([Hu et al. 2008](http://yifanhu.net/PUB/cf.pdf)). Mô hình của Hu không có chỗ cho trường hợp "đã quan sát nhưng không thích". Nếu áp nguyên mô hình đó, một review False kèm 300 giờ chơi sẽ thành một positive có độ tin cậy cao, điều này sai với dữ liệu Steam. Vì vậy `hours` chỉ nên nhân vào nhãn quan sát được. YouTube cho thấy dislike **chỉ có tác dụng khi dùng làm nhãn huấn luyện**. Dùng cả làm feature lẫn nhãn giảm dislike rate 2,44%, còn chỉ làm feature thì giảm 0,34%, không có ý nghĩa thống kê. YouTube cũng chọn một số hạng riêng `−w·log(1−p)` thay vì nhét negative vào mẫu số softmax ([Wang et al. 2023](https://arxiv.org/abs/2308.12256)). Hai hướng này là hai cách hợp lệ để đưa False vào InfoNCE ở Pha 3. Hybrid+BPR phải dùng cùng nguồn negative với Pha 3. Với BPR, BPR++ và multi-channel BPR có tiền lệ cho cặp phân bậc pos ≻ unknown ≻ explicit-neg ([Lerche & Jannach 2014](https://web-ainf.aau.at/pub/jannach/files/Conference_RecSys2014a.pdf); [Loni 2016](https://research.tudelft.nl/en/publications/bayesian-personalized-ranking-with-multi-channel-user-feedback), chưa đọc toàn văn). Với `hours`, nghiên cứu duration bias cho thấy **nhãn theo thời lượng làm mô hình thiên về item dài**. Cách khắc phục là đổi thời lượng sang quantile trong từng bucket ([Zhan et al. 2022](https://arxiv.org/abs/2206.06003)). Trên Steam, tương tự là MMO hay sandbox tích lũy hàng trăm giờ. Phương án thay thế là percentile giờ chơi trong từng game, fit trên review ≤ cutoff. Bằng chứng duy nhất trên Steam rằng nhãn explicit "không thay đổi nhiều" kết quả ALS đến từ một dự án sinh viên, và mới chỉ thấy qua snippet ([Batra et al.](https://arxiv.org/pdf/2305.04890)). Cũng vì vậy nên giữ một ablation không dùng explicit negative.

## Content chủ yếu giúp item lạnh, tags và description cần so trực tiếp với nhau

LightFM cho thấy hybrid content **thắng ở cold-start và dữ liệu thưa, còn với item nhiều tương tác thì "ít nhất ngang" MF** ([Kula 2015](https://arxiv.org/abs/1507.08439)). DropoutNet, Heater và CLCRec đều bổ sung cơ chế buộc phần content phải mang thông tin khi thiếu ID ([DropoutNet](https://proceedings.neurips.cc/paper/2017/file/dbd22ba3bd0df8f385bdac3e9f8be207-Paper.pdf); [CLCRec](https://arxiv.org/abs/2107.05315)). Vì vậy Pha 2 phải báo cáo kết quả theo bucket độ phổ biến của item. Nếu thêm dropout `V[i]` hoặc loss căn chỉnh `V[i]`–`W c_i`, phải thêm ở cả Pha 2 lẫn Pha 3. Về encoder, beeFormer cho thấy sentence encoder dùng nguyên **chỉ xấp xỉ ngang CF đơn giản trong split thời gian**: Recall@20 của all-mpnet là 0,0218, của bge-m3 là 0,0398, của ALS là 0,0344 trên Amazon Books ([Vančura et al. 2024](https://arxiv.org/html/2409.10309v2)). UniSRec phải whitening embedding BERT trước khi dùng ([Hou et al. 2022](https://arxiv.org/pdf/2206.05941)). Trên Steam, một luận văn Tartu báo TF-IDF thắng SBERT ([UT DSpace](https://dspace.ut.ee/bitstreams/a78db95c-0396-4f7c-9e1c-671e8d619ce8/download), **chỉ snippet, chưa kiểm chứng**). GameRecs dùng tags có trọng số vote và bỏ tag chế độ chơi (**chỉ snippet**). Từ đó, ablation hợp lý là ba biến thể tags-only, description-only và tags+description, mỗi biến thể chạy với TF-IDF+SVD và một encoder đóng băng. Ở Pha 3, semi-hard negative theo content có rủi ro false negative cao. SRNS ghi nhận false negative có điểm cao và ổn định ([Ding et al. 2020](https://arxiv.org/abs/2009.03376v1)). Robinson et al. cho thấy tăng độ khó không giúp khi nhãn chỉ gần đúng ([ICLR 2021](https://arxiv.org/abs/2010.04592)). Vì vậy semi-hard chỉ là ablation: lấy mẫu trong một dải tương đồng, loại game cùng franchise và trộn với negative ngẫu nhiên.

## Mọi trường tổng hợp đến lúc thu thập đều là rò rỉ

Kaufman et al. định nghĩa leakage là dùng thông tin "không hợp lệ" tại thời điểm dự đoán. Họ coi việc dùng dữ liệu IMDB hiện tại "không roll-back" là **"simple leakage"**. Theo họ, **chọn feature hoặc hyperparameter bằng dữ liệu sau điểm tách cũng là leakage**, và đây là dạng nguy hiểm nhất ([Kaufman et al. 2011](https://www.cs.umb.edu/~ding/history/470_670_fall_2011/papers/cs670_Tran_PreferredPaper_LeakingInDataMining.pdf)). Như vậy `positive_ratio`, `rating` hay `user_reviews` phải bị loại ngay cả khỏi bước phân tích chọn cấu hình. Sun gọi việc đếm popularity bằng tương tác tương lai là "unrealistic" ([Sun 2023](https://ar5iv.labs.arxiv.org/html/2210.04149)). Ji et al. cho thấy rò rỉ do bỏ qua dòng thời gian toàn cục **làm thứ hạng các mô hình đảo lộn khó đoán** ([TOIS 2023](https://arxiv.org/abs/2010.11060v4)). Cách thay thế hợp lệ là tính lại popularity hoặc tỉ lệ positive chỉ từ review ≤ cutoff. Về đánh giá, chấm trên mẫu negative có thể đảo kết quả so sánh. Ví dụ, mô hình A hơn B 8,1% recall@20 khi chấm full ranking nhưng thua 4,7% khi chấm với 100 mẫu ([Hidasi & Czapp 2023](https://ar5iv.labs.arxiv.org/html/2307.14951)). Krichene & Rendle khuyên tránh sampled metric ([KDD 2020](https://research.google/pubs/on-sampled-metrics-for-item-recommendation/), chỉ đọc abstract). Với description và tags, không có cách roll-back. Theo Kaufman, trường hợp này chỉ "bịt được một phần". Báo cáo nên nêu rõ giới hạn snapshot, và chỉ khẳng định về cold-start cho game ra mắt giữa `T1` và `T2`.

## Những điểm còn chưa chắc chắn

Điểm quan trọng nhất là **nghĩa thời gian của `hours` không có trong tài liệu**. Qua API, Kaggle card không có mô tả cột hay ngày snapshot nào. Card chỉ ghi phiên bản 28, cập nhật ngày 2024-08-14, với tần suất cập nhật "never" ([Kaggle API](https://www.kaggle.com/api/v1/datasets/view/antonkozyriev/game-recommendations-on-steam)). Trang web của Kaggle cần JavaScript nên chưa xem được bảng mô tả cột. Mô tả "hours of playtime the user has on the game" chỉ đến từ snippet của một repo bên thứ ba (**chưa kiểm chứng**). Cách kiểm tra thực nghiệm là đếm các review có `hours` vượt số giờ có thể chơi giữa `date_release` và `date`. Các điểm chưa chắc chắn khác: Steam gán nhãn `rating` theo quy tắc nào chưa được xác minh từ Valve. CPGRec (WWW 2024) dùng bộ Steam nào chưa rõ. Kết quả của Gong et al. (2020), một hybrid Steam dùng giá, ngày phát hành và genre, mới chỉ thấy qua abstract. Kết quả về α và trọng số hours trên Steam mới đến từ một preprint chưa bình duyệt. Chưa tìm thấy nghiên cứu nào đo explicit negative trên Steam dưới split thời gian. Cũng chưa có nghiên cứu nào đánh giá cờ nền tảng hay `steam_deck` như đặc trưng gợi ý.

## Kết luận

Câu hỏi về trường dữ liệu thực ra là câu hỏi về giao thức. Một trường chỉ hợp lệ khi giá trị của nó có thể tái dựng tại `T1` và `T2`. Trong 25 trường, chỉ có ID, nhãn, ngày và content văn bản đạt yêu cầu này, và content văn bản chỉ đạt khi chấp nhận giới hạn snapshot. Vì vậy phạm vi Pha 2 hẹp hơn tên "Hybrid CF + content" gợi ý. Thực chất đây là phép thử xem tags và description có cải thiện item thưa hoặc lạnh hay không. Pha 3 chỉ có ý nghĩa nếu giữ nguyên tập trường và nguồn negative của Pha 2.

Việc nên làm trước khi train là kiểm tra `hours` theo ngày phát hành. Nếu dữ liệu cho thấy giờ chơi được đo lúc thu thập, ablation trọng số hours nên được hạ xuống thành phân tích phụ có cảnh báo rò rỉ, không trình bày như một cải tiến mô hình.

## Tài liệu tham khảo

- Kozyriev, *Game Recommendations on Steam*, Kaggle (CC0, v28 2024-08-14): https://www.kaggle.com/datasets/antonkozyriev/game-recommendations-on-steam · metadata API: https://www.kaggle.com/api/v1/datasets/metadata/antonkozyriev/game-recommendations-on-steam
- Kaufman, Rosset & Perlich, *Leakage in Data Mining*, KDD 2011: https://www.cs.umb.edu/~ding/history/470_670_fall_2011/papers/cs670_Tran_PreferredPaper_LeakingInDataMining.pdf
- Ji, Sun, Zhang & Li, *A Critical Study on Data Leakage in Recommender System Offline Evaluation*, TOIS 2023: https://arxiv.org/abs/2010.11060v4
- Ji et al., *A Re-visit of the Popularity Baseline*, SIGIR 2020: https://arxiv.org/abs/2005.13829
- Sun, *Take a Fresh Look at Recommender Systems from an Evaluation Standpoint*, SIGIR 2023: https://ar5iv.labs.arxiv.org/html/2210.04149
- Meng et al., *Exploring Data Splitting Strategies…*, RecSys 2020: https://arxiv.org/pdf/2007.13237
- Hidasi & Czapp, *Widespread Flaws in Offline Evaluation of Recommender Systems*, RecSys 2023: https://ar5iv.labs.arxiv.org/html/2307.14951
- Krichene & Rendle, *On Sampled Metrics for Item Recommendation*, KDD 2020: https://research.google/pubs/on-sampled-metrics-for-item-recommendation/
- Cañamares & Castells, RecSys 2020: https://castells.github.io/papers/recsys2020.pdf
- Hu, Koren & Volinsky, *Collaborative Filtering for Implicit Feedback Datasets*, ICDM 2008: http://yifanhu.net/PUB/cf.pdf
- Rendle et al., *BPR*, UAI 2009: https://arxiv.org/pdf/1205.2618
- Wang et al., *Learning from Negative User Feedback…*, RecSys 2023: https://arxiv.org/abs/2308.12256
- Lerche & Jannach, *BPR++*, RecSys 2014: https://web-ainf.aau.at/pub/jannach/files/Conference_RecSys2014a.pdf
- Loni et al., *Multi-channel BPR*, RecSys 2016: https://research.tudelft.nl/en/publications/bayesian-personalized-ranking-with-multi-channel-user-feedback
- Zhan et al., *Deconfounding Duration Bias in Watch-time Prediction*, KDD 2022: https://arxiv.org/abs/2206.06003
- Kang & McAuley, *SASRec*, ICDM 2018: https://arxiv.org/pdf/1808.09781
- Wan & McAuley, *chainRec*, RecSys 2018: https://cseweb.ucsd.edu/%7Ejmcauley/pdfs/recsys18b.pdf
- Pathak, Gupta & McAuley, SIGIR 2017: https://cseweb.ucsd.edu/~jmcauley/pdfs/sigir17.pdf
- FairGap, arXiv 2608.08284 (2026): https://arxiv.org/html/2608.08284v1
- sunaggn/steam-nffmtl: https://github.com/sunaggn/steam-nffmtl
- Batra et al., *Steam Recommendation System*, arXiv 2305.04890 (chỉ snippet): https://arxiv.org/pdf/2305.04890
- Kula, *LightFM*, 2015: https://arxiv.org/abs/1507.08439
- Volkovs et al., *DropoutNet*, NeurIPS 2017: https://proceedings.neurips.cc/paper/2017/file/dbd22ba3bd0df8f385bdac3e9f8be207-Paper.pdf
- Wei et al., *CLCRec*, ACM MM 2021: https://arxiv.org/abs/2107.05315
- Hou et al., *UniSRec*, KDD 2022: https://arxiv.org/pdf/2206.05941
- Li et al., *Recformer*, KDD 2023: https://arxiv.org/pdf/2305.13731
- Vančura et al., *beeFormer*, RecSys 2024: https://arxiv.org/html/2409.10309v2
- GameRecs (chỉ snippet): https://homepages.tuni.fi/konstantinos.stefanidis/docs/qauca19.pdf
- Luận văn Tartu, TF-IDF vs SBERT trên Steam (chỉ snippet): https://dspace.ut.ee/bitstreams/a78db95c-0396-4f7c-9e1c-671e8d619ce8/download
- Huang, Oosterhuis & de Rijke, *DANCER*, WSDM 2022: https://arxiv.org/abs/2111.12481
- Ding et al., *SRNS*, NeurIPS 2020: https://arxiv.org/abs/2009.03376v1
- Robinson et al., *Hard Negative Samples*, ICLR 2021: https://arxiv.org/abs/2010.04592
- PC Gamer, động lực giá Steam (báo chí): https://www.pcgamer.com/how-long-does-it-take-games-to-be-cheap-on-steam/3
- Tilburg, luận văn về discount: https://arno.uvt.nl/show.cgi?fid=161709
