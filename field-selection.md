# Lựa chọn trường dữ liệu theo từng pha

Ba tiêu chí chọn trường:

1. **Đã biết tại cutoff**: trường chứa thông tin sinh ra sau cutoff bị loại (chống rò rỉ).
2. **Đúng câu hỏi của pha**: pha 1 đo tín hiệu cộng tác, pha 2 đo content, pha 3 đo loss.
3. **So sánh công bằng**: mỗi pha chỉ thêm một thứ so với pha trước.

## Pha 1 — MF / CF

| Trường | Vai trò |
|---|---|
| `user_id`, `app_id` | Chỉ số dòng của `U`, `V`. Không dùng đặc trưng khác để đo riêng tín hiệu cộng tác |
| `is_recommended` | Nhãn: True = positive, False = explicit negative (đáng tin hơn negative lấy mẫu) |
| `date` | Chia theo `T1`/`T2`, giữ review mới nhất mỗi cặp, lịch sử ≤ cutoff |
| `review_id` | Khóa duy nhất, phân định khi trùng ngày |
| `date_release` | Lọc tập ứng viên tại cutoff (không phải đặc trưng) |
| `hours` | Tùy chọn: trọng số tin cậy, ablation sau baseline. Cần kiểm tra là giờ lúc review hay lúc thu thập |

Không dùng:
- **Description/tags**: để dành cho pha 2.
- **`helpful`, `funny`**: vote của người khác, đếm đến lúc thu thập (rò rỉ); đo độ hữu ích của review, không phải sở thích.
- **`users.products`, `users.reviews`**: số đếm lúc thu thập, không phải danh sách game.

## Pha 2 — Hybrid CF + content

Giữ trường pha 1, thêm:

| Trường | Vai trò |
|---|---|
| `description` + `tags` → `c_i` | Vector nội dung, chiếu qua `W` để tạo `g_i = V[i] + W c_i`. Giúp game ít/không có review. Chỉ 2,42% game thiếu cả hai |
| `win`/`mac`/`linux` | Tùy chọn; ổn định nhưng ít thông tin |

Không dùng:
- **`positive_ratio`, `rating`, `user_reviews`**: tổng hợp từ mọi review, kể cả giai đoạn test (rò rỉ nặng nhất) và gây popularity bias.
- **`price_*`, `discount`**: giá tại lúc thu thập, thay đổi theo thời gian.
- **`steam_deck`**: nhãn gán sau 2022, không đúng với review cũ hơn.
- **`title`**: gần như định danh, ít nghĩa hơn description; có thể thử ablation.

Giới hạn: description/tags là bản chụp hiện tại, không phải tại từng cutoff.

## Pha 3 — Hybrid + InfoNCE

Không thêm trường: cùng dữ liệu và kiến trúc pha 2, chỉ đổi loss. Content dùng thêm trong ablation semantic semi-hard negative, tránh game quá giống positive vì dễ là false negative.

## Tài liệu tham khảo

| Lựa chọn | Paper |
|---|---|
| Chia theo thời gian | Ji et al., *A Critical Study on Data Leakage in Recommender System Offline Evaluation*, TOIS 2023; Meng et al., *Exploring Data Splitting Strategies…*, RecSys 2020 |
| Loại trường rò rỉ | Kaufman et al., *Leakage in Data Mining*, KDD 2011 / TKDD 2012 |
| Chấm toàn catalog | Krichene & Rendle, *On Sampled Metrics for Item Recommendation*, KDD 2020 |
| Nhãn + trọng số `hours` | Hu, Koren & Volinsky, *Collaborative Filtering for Implicit Feedback Datasets*, ICDM 2008 |
| Content cho item | Kula, *LightFM*, 2015; He & McAuley, *VBPR*, AAAI 2016; Wang & Blei, *CTR*, KDD 2011 |
| ID dropout | Volkovs et al., *DropoutNet*, NeurIPS 2017 |
| Tránh false negative | Ding et al., *Simplify and Robustify Negative Sampling*, NeurIPS 2020 |
| In-batch negative bị thiên lệch | Yi et al., *Sampling-Bias-Corrected Neural Modeling…*, RecSys 2019 |
| Contrastive + cold start | Wei et al., *CLCRec*, ACM MM 2021 |
