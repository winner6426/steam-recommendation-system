# Kế hoạch dự án: Steam Game Recommender

## 1. Mục tiêu và phạm vi

Xây dựng hệ thống xếp hạng Top-K game cho user từ lịch sử review Steam. Dự án tiến hành ba phase có thể so sánh: **(1) CF bằng Matrix Factorization, (2) CF + content, (3) hybrid + contrastive learning**. [architecture.md](architecture.md) là đặc tả luồng dữ liệu, công thức và phục vụ tương ứng.

Câu hỏi nghiên cứu:

1. MF/CF có vượt mốc Popularity khi gợi ý game chưa được user review không?
2. Thêm description/tags có cải thiện Top-K và cold item không?
3. InfoNCE với nhiều negative có cải thiện so với hybrid train bằng BCE/BPR không?

Kết quả Phase 3 không được giả định trước. Chọn mô hình/cấu hình theo validation trên giao thức cố định; test chỉ đánh giá lựa chọn đó, kèm phân tích trade-off về tốc độ và cold start.

## 2. Dataset và giới hạn cần nhớ

| Tệp | Dữ liệu dùng | Lưu ý |
|---|---|---|
| `data/raw/recommendations.csv` | `user_id`, `app_id`, `date`, `is_recommended`, `hours`, `review_id` | Review `False` là negative xác nhận cho game đó; không có review là unknown |
| `data/raw/games.csv` | `app_id`, `title`, `date_release`, platform | `positive_ratio`, `rating`, `user_reviews` có nguy cơ rò rỉ thông tin thời gian |
| `data/raw/games_metadata.json` | `app_id`, `description`, `tags` | 20,39% thiếu description; 2,42% thiếu cả description và tags |
| `data/raw/users.csv` | `user_id`, `products`, `reviews` | `products` và `reviews` là **số lượng**, không phải danh sách game hoặc nội dung review |

Theo `data/dataset_report.txt`, có 41.154.794 review, trong đó 35.304.398 `True` và 5.850.396 `False`. Có 13.781.059 user và 37.610 game xuất hiện trong review; catalog có 50.872 game. Trung bình số review trên mỗi user có review xấp xỉ 3, nên phải báo cáo riêng hiệu quả với user ít lịch sử. Báo cáo hiện còn ghi đường dẫn cũ `archive/` bên trong; file dữ liệu thực tế đang nằm trong `data/raw/`.

Dataset **không có** danh sách toàn bộ game user đã sở hữu/chơi, click, session hoặc thuộc tính nhân khẩu học. Hệ thống có thể loại game user đã review, nhưng không thể cam kết loại toàn bộ game họ từng chơi. Không sử dụng Knowledge Graph hay session-based recommendation trong phạm vi chính.

## 3. Công việc chung trước ba phase

### 3.1. Tạo dữ liệu sạch có thể tái lập

- Hoàn tất kiểm tra toàn bộ bốn nguồn và xác định quy tắc xử lý bất thường theo [architecture.md, mục 3.1](architecture.md#31-điều-kiện-nạp-toàn-bộ-và-lưu-trữ-preprocess), rồi **nạp tất cả dữ liệu để preprocess toàn bộ**. Luồng chính không giới hạn dòng hoặc lấy mẫu; smoke là chế độ kiểm thử tùy chọn. Giữ các file gốc trong `data/raw/` nguyên trạng.
- Triển khai `scripts/prepare_data.py` với `configs/data.json`, package/dependencies trong `pyproject.toml` và các module `src/steamrec/data/{io,clean,validate,split,manifest}.py`. Theo thiết kế mục 3 của `architecture.md`: `load_file` đọc cả bốn nguồn, clean và validate trước, chỉ sau đó lưu các bảng `interactions`, `games`, `users`, `content` vào database bền vững. Không lưu bảng raw, không thêm hậu tố `_clean`, không yêu cầu thư mục `run_*`.
- Lưu bốn bảng riêng, không ghép toàn bộ cột thành một bảng interaction lớn. Dữ liệu interactions giữ review lặp; bước snapshot áp dụng dedup theo cutoff tại mục 3 của `architecture.md`, rồi tạo mapping ID và artifact phục vụ train/evaluate. `split.py` hiện có helper dedup; phần tạo snapshot đầy đủ chờ chọn `T1/T2` từ báo cáo tháng.
- Giữ Parquet cuối sau preprocess theo đường dẫn config; bảo toàn đầy đủ cột nguồn và thứ tự 8/13/3 của recommendations/games/users theo schema tại `architecture.md`, không thêm cờ vào ba bảng này. Test schema và giá trị các cột đã giữ, kiểm tra số dòng Parquet cuối; các trường có nguy cơ leakage được lưu nhưng không tự động dùng làm feature.
- Đối chiếu số dòng cả bốn nguồn với dữ liệu giữ lại/cách ly, mở lại database xác nhận bảng và số dòng. Ghi SHA-256 input, quy tắc xử lý, audit join/ngày/content, thống kê tháng và artifact vào manifest theo đường dẫn config. Kiểm thử pipeline trên fixture có nhãn trái nhau, ngày sai, ID trùng và content thiếu trước khi bàn giao mã; báo riêng việc preprocess dữ liệu thật đã chạy hay chưa.
- Không gán `0` cho tất cả ô trống của ma trận user-game. `hours` có thể thử làm confidence weight sau baseline; không thay thế nhãn `is_recommended`.

**Hoàn thành bước preprocess khi:** cả bốn nguồn đã được nạp đầy đủ vào bảng bền vững, dữ liệu sạch và báo cáo được tạo trên toàn bộ input, kiểm tra số dòng và khả năng mở lại database đạt yêu cầu, có manifest tái lập. Đây là tiêu chí hoàn thành cần thực hiện, không phải khẳng định pipeline hiện đã chạy toàn bộ.

### 3.2. Khóa split và evaluator

- Chọn hai mốc `T1 < T2` từ phân bố ngày review. Validation: train bằng review `date ≤ T1`, dự đoán positive `(T1,T2]`; chọn hyperparameter tại đây. Test: retrain **từng model với cùng cấu hình đã chọn** bằng review `date ≤ T2`, rồi dự đoán positive `date > T2`. Không dùng test để chỉnh cấu hình.
- Ở mỗi cutoff, chỉ dùng dữ liệu đã có để tính popularity, mapping ID, lịch sử user và negative. User đủ điều kiện có ít nhất một positive trước/đúng cutoff và một positive hợp lệ sau cutoff; đánh giá **mọi user đủ điều kiện**, không dùng cohort mẫu làm kết quả cuối.
- Với mỗi user, chấm điểm **toàn bộ catalog hợp lệ tại cutoff** cho mọi model: game đã phát hành trước/đúng cutoff, chưa được user review đến cutoff. Không lọc theo review tương lai. Positive target phát hành sau cutoff hoặc không có ngày phát hành hợp lệ không thể xếp hạng ở snapshot đó; báo cáo số bị loại.
- Tính `Recall@10/20`, `NDCG@10/20`, thêm HitRate@K và coverage. Popularity dùng cùng snapshot/candidate/evaluator. Chấm điểm theo batch, giữ Top-K, không tạo ma trận user-game đặc.

### 3.3. Khóa chiến lược negative

Thí nghiệm chính ghép mỗi positive `(u,p)` với: (a) game user đã review `False` trước cutoff, nếu có; (b) game chưa review lấy ngẫu nhiên từ catalog hợp lệ. Nguồn (b) chỉ là **mẫu âm giả định**, cho trọng số thấp hơn review `False`. Không sử dụng review tương lai để loại/chứng thực negative. Giữ cùng định nghĩa sampler, tỷ lệ nguồn và seed cho các model đối chứng; sinh mẫu xác định được theo snapshot/epoch/positive thay vì bắt buộc lưu toàn bộ negative pool trên đĩa.

Semantic semi-hard negative theo content là **ablation riêng sau baseline**: thử vùng tương đồng vừa phải, tránh game giống positive nhất vì nguy cơ false negative. Nếu so tác dụng kiến trúc/loss ở ablation này, chạy lại các đối chứng với cùng sampler semantic. Vùng top 5–20%, số lượng negative, tỷ lệ trộn và trọng số là hyperparameter chọn bằng validation; không trộn metric ablation với bảng chính.

**Hoàn thành phần chung khi:** có định nghĩa hai snapshot và dữ liệu theo cutoff, sampler tái lập được từ config/seed, thống kê sau xử lý, manifest và evaluator full-catalog chạy được với Popularity trên tất cả user đủ điều kiện.

## 4. Phase 1 — CF bằng Matrix Factorization

### Model và huấn luyện

Tạo hai bảng `U[num_users, d]`, `V[num_games, d]`, thử `d=64` trước. Điểm là `s1(u,i)=U[u]·V[i]/sqrt(d)`. Mỗi bước: lấy batch `(u,i,label,weight)`, tính score, tính loss, backpropagation rồi cập nhật `U,V` bằng optimizer. Các ô user-game không có review không được xem là nhãn thật 0.

Train ít nhất hai cấu hình **cùng model MF**:

- `MF+BCE`: weighted binary cross entropy trên positive, explicit negative và sampled unknown; review `False` đáng tin hơn sampled unknown.
- `MF+BPR`: với `(u,p,n)`, tối ưu `−log σ(s1(u,p)−s1(u,n))` để trực tiếp học thứ tự Top-K.

Popularity là baseline không cần train embedding. Chọn learning rate, regularization, embedding dimension và thời điểm dừng bằng validation. Dùng PyTorch `nn.Embedding`; chạy smoke test local trên tập nhỏ để phát hiện lỗi, nhưng job kết quả cuối phải dùng **toàn bộ review hợp lệ** trong cửa sổ train của snapshot trên Kaggle/máy khác. Với 13,8 triệu user, bảng embedding ID toàn bộ có thể tốn nhiều GB trước cả gradient/optimizer state: ước tính RAM/VRAM, đĩa tạm và thời gian từ benchmark nhỏ trước khi chạy; đọc dữ liệu theo batch và lưu checkpoint có thể tiếp tục sau gián đoạn. Không khẳng định một môi trường Kaggle bất kỳ đủ tài nguyên trước khi đo.

### Kiểm chứng và đầu ra

- Chứng minh loss giảm và điểm positive đã biết thường cao hơn negative trên validation.
- Đo Top-K cho Popularity, `MF+BCE`, `MF+BPR` trên cùng evaluator; validation dùng snapshot `T1`, test dùng model retrain đến `T2`.
- Lưu checkpoint, mapping ID theo snapshot, config/seed, số dòng train/user test, thời gian/bộ nhớ chạy, bảng metric và ví dụ Top-K đã loại game review đến cutoff.

**Câu trả lời Phase 1:** chỉ dùng collaborative signal có đủ mạnh làm baseline không? MF chưa xử lý được game/user hoàn toàn mới bằng ID embedding.

## 5. Phase 2 — Thêm content vào CF

### Tạo feature nội dung

Ghép description và tags của game thành văn bản, xử lý giá trị thiếu. Bắt đầu với TF-IDF + TruncatedSVD hoặc một Sentence Transformer có sẵn để có vector dense `c_i`/game; lưu cùng version encoder. Nếu fit TF-IDF/SVD, chỉ dùng game đủ điều kiện tại cutoff của snapshot; encoder pretrained chỉ dùng suy luận. Metadata chỉ là snapshot hiện tại nên vẫn phải ghi rõ giới hạn lịch sử. Có thể thêm platform ổn định; tạm tránh các trường tổng hợp review hoặc giá/discount thay đổi theo thời gian trong thí nghiệm chính.

### Model và huấn luyện

Giữ `U[u]`, `V[i]` từ cấu trúc Phase 1, thêm lớp chiếu `W`:

```text
g_i = V[i] + W c_i
s2(u,i) = U[u] · g_i / sqrt(d)
```

Train `U,V,W` với **cùng snapshot, định nghĩa sampler chính và BCE** của Phase 1; điều này cho phép so sánh `MF+BCE` với `Hybrid+BCE`. Nếu khởi tạo từ checkpoint Phase 1 rồi fine-tune, cho baseline Phase 1 train tiếp cùng số bước. **Bắt buộc chạy `Hybrid+BPR`** để có đối chứng ranking trước Phase 3. Thử ID dropout để nhánh content học cách dự đoán khi game không có embedding từ tương tác.

### Kiểm chứng và đầu ra

- Báo cáo `Hybrid+BCE − MF+BCE` và `Hybrid+BPR − MF+BPR` trên cùng user/candidate/metric.
- Đánh giá riêng nhóm game ít review và **cold item giả lập**: giấu interaction train của một nhóm game nhưng giữ description/tags, rồi đo khả năng tìm positive test của nhóm này.
- Lưu checkpoint hybrid, text encoder/version, content vectors và bảng ablation có/không có content.

**Câu trả lời Phase 2:** content cải thiện kết quả nào, và có giúp game mới/ít review không? Game thiếu cả description/tags cần fallback; cold user chưa được giải quyết bởi game content tower.

## 6. Phase 3 — Hybrid với contrastive learning

### Model và huấn luyện

Giữ **nguyên** `U[u]` và `g_i=V[i]+Wc_i` của Phase 2. Với một positive `p` và nhiều negative `N`, tối ưu:

```text
q(u,i) = s2(u,i) / τ
L = -log( exp(q(u,p)) /
          [exp(q(u,p)) + Σ(n∈N) w_n exp(q(u,n))] )
```

Thử `τ` và trọng số từng nguồn negative trên validation. Không yêu cầu mỗi user phải có explicit negative. Giữ nguồn sampler chính tương đương với Phase 2; nếu bổ sung số lượng mẫu, chạy đối chứng để tách tác dụng của sampler khỏi tác dụng của loss. Không dùng in-batch negatives chưa được lọc như một nhãn âm chắc chắn.

InfoNCE với một negative và trọng số 1 có dạng tương đương BPR sau khi thay thang điểm; vì vậy thí nghiệm chính phải dùng nhiều negative và so sánh với `Hybrid+BPR` đủ mạnh.

### Kiểm chứng và đầu ra

- Cùng evaluator, so sánh `Hybrid+InfoNCE` với `Hybrid+BCE` và `Hybrid+BPR` trên warm/sparse user, warm/cold item.
- Ablation riêng: explicit only khi có dữ liệu, semantic+random, mixed; thử khoảng similarity và temperature. Khi đổi sampler, chạy đối chứng Hybrid+BCE/BPR tương ứng. Báo cáo cả trường hợp Phase 3 không thắng.
- Lưu checkpoint, seed và phiên bản sampler, hyperparameter, manifest dữ liệu/snapshot và bảng metric cuối; không yêu cầu lưu vật lý mọi cặp negative.

**Câu trả lời Phase 3:** học tương phản với nhiều negative có cải thiện Top-K trên cùng kiến trúc hybrid không?

## 7. Cold start và bản demo

| Tình huống | Cách xử lý trong MVP |
|---|---|
| User đã có embedding trong bundle | Dùng embedding user của model được chọn; loại game đã review đến cutoff của bundle |
| User mới không chọn gì | Popularity mặc định |
| User mới chọn 2–3 game/tags | Gợi ý theo content của game/tags đã chọn; không giả định đó là vector user MF |
| Game chưa có ID embedding, có content | Phase 2/3 dùng `g_i=Wc_i`; MF không có nhánh này nên cần fallback |
| Game thiếu cả content và review | Fallback theo catalog/metadata phù hợp; ghi rõ hạn chế |

Nếu muốn user mới được chấm bằng chính hybrid model, cần bổ sung user history encoder hoặc quy trình fold-in và đánh giá riêng. Dữ liệu hiện tại không có demographic, click hoặc session để huấn luyện các hướng đó. Không bắt user đánh giá các game họ chưa biết; cho họ chọn game/tags quen thuộc và vẫn có đường bỏ qua.

Sau khi chọn model, xuất **bundle phục vụ có version**, tách khỏi checkpoint train: user/item embeddings hoặc tham số tạo chúng, ánh xạ ID, catalog, content vectors/encoder tương ứng, lịch sử game đã review để lọc, popularity fallback và manifest cutoff/config/seed/phiên bản dữ liệu. Demo local đọc bundle thay vì quét lại CSV gốc; user embeddings lớn có thể đọc từng phần từ đĩa, chấm điểm trực tiếp catalog. Giao diện đơn giản và FastAPI chỉ là lựa chọn khi làm demo, chưa phải phần của ba thí nghiệm. FAISS, PostgreSQL/pgvector hoặc Qdrant chỉ bổ sung khi có số đo tốc độ/lưu trữ chứng minh cần thiết.

## 8. Bảng kết quả cuối cần bàn giao

| Model | Recall@10 | NDCG@10 | Recall@20 | Cold-item Recall@10 | Ghi chú |
|---|---:|---:|---:|---:|---|
| Popularity | | | | | Không cá nhân hóa |
| MF+BCE | | | | | CF baseline |
| MF+BPR | | | | | CF ranking baseline |
| Hybrid+BCE | | | | | Đo đóng góp của content |
| Hybrid+BPR | | | | | Đối chứng ranking cho Phase 3 |
| Hybrid+InfoNCE | | | | | Proposed model; kết quả cần kiểm chứng |

Điều kiện bàn giao: bảng chính dùng cùng giao thức hai snapshot, sampler chính và evaluator full-catalog; train trên toàn bộ review hợp lệ đến từng cutoff và đánh giá tất cả user đủ điều kiện. Có config/seed và manifest để tái chạy, số lượng dữ liệu sau lọc, số user/target không đủ điều kiện, thời gian/bộ nhớ chạy, phân tích cold start và hạn chế của review-only dataset. Bảng semantic semi-hard ablation tách riêng. Demo local phải đọc được bundle đã xuất và xử lý user/game mới theo fallback đã nêu. Không tuyên bố hybrid hay contrastive hiệu quả hơn nếu metric chưa chứng minh.
