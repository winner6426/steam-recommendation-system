# Quy tắc làm việc cho agent — Steam Recommendation System

Áp dụng trong thư mục `steam-recommendation-system/`. Trước khi sửa pipeline hoặc mô hình, đọc `architecture.md` (thiết kế và giao thức) và `project.md` (công việc, điều kiện hoàn thành). Không sao chép thông số thí nghiệm giữa nhiều tài liệu; khi thay đổi một quyết định, cập nhật tài liệu sở hữu quyết định đó và các tham chiếu liên quan. Yêu cầu trực tiếp của người dùng có ưu tiên cao hơn tài liệu; nêu rõ nếu yêu cầu đó thay đổi phạm vi hoặc giao thức nghiên cứu.

## Phạm vi và an toàn dữ liệu

- Chỉ làm việc được yêu cầu. Ba phase chính là MF/CF, Hybrid CF + content, và cùng Hybrid với InfoNCE; đây là ba thí nghiệm so sánh, không phải cascade phục vụ. Không tự thêm LLM, Knowledge Graph, session model, hạ tầng phân tán hay frontend đầy đủ.
- File CSV/JSON gốc nằm trực tiếp trong `data/`; không sửa, ghi đè, xóa hoặc di chuyển chúng. Không xóa checkpoint, kết quả, notebook hoặc thay đổi có sẵn của người dùng. Đầu ra sinh mới dùng vị trí và quy tắc Git mô tả trong `architecture.md`.
- Không tự chạy job train toàn bộ hoặc tạo artifact lớn khi yêu cầu chỉ là tài liệu/mã nguồn. Khi được yêu cầu train, chạy smoke test trước, ước tính tài nguyên và chỉ chạy job lớn trên môi trường đã được chọn; báo rõ phần nào thực sự đã chạy.

## Bất biến dữ liệu và đánh giá

- `is_recommended=True` là positive; `False` là explicit negative chỉ cho game đã review; chưa review là unknown. `users.csv` không chứa danh sách game sở hữu/chơi. Không tuyên bố đã loại hết game user từng chơi.
- Tuân thủ giao thức hai snapshot trong `architecture.md`: không dùng review tương lai để tạo lịch sử, fit mapping/popularity/sampler, lọc game hay chọn hyperparameter. Validation chọn cấu hình với dữ liệu đến `T1`; test retrain đến `T2` và chỉ báo cáo kết quả, không chọn lại cấu hình.
- Kết quả chính train trên toàn bộ review hợp lệ đến từng cutoff, đánh giá full-catalog cho **mọi user đủ điều kiện** bằng cùng định nghĩa candidate/evaluator. Smoke test hoặc sampled ranking không được trình bày như kết quả chính.
- Sampler chính dùng explicit `False` và unknown ngẫu nhiên, tái lập từ config/seed; semantic semi-hard là ablation riêng. Khi thay sampler, số negative hoặc ngân sách train, chạy đối chứng tương ứng. `Hybrid+BPR` là đối chứng bắt buộc của Phase 3.
- Tránh feature tổng hợp từ review tương lai (`positive_ratio`, `rating`, `user_reviews`, `helpful`, `funny`). Ghi rõ giới hạn snapshot của description/tags và báo cáo cold-start đúng phạm vi bằng chứng.

## Mã nguồn, kiểm chứng và bàn giao

- Logic split, sampler, metrics và lọc game dùng chung giữa các phase; Phase 3 dùng lại kiến trúc Hybrid. Đường dẫn, cutoff, seed và hyperparameter thuộc config, không hard-code đường dẫn máy cá nhân. Không commit dữ liệu lớn, checkpoint hay secret.
- Lưu manifest nhỏ về dữ liệu, cutoff, sampler, config/seed và artifact để tái chạy; `data/` hiện bị Git bỏ qua nên manifest cần theo dõi phiên bản không được chỉ nằm trong `data/`.
- Test các điểm dễ sai: split thời gian, unseen ID, negative sampling, lọc game đã review và Recall/NDCG. Trước khi báo hoàn thành, chạy kiểm chứng phù hợp; nêu rõ phần chưa chạy, file đã thay đổi, tác động và giới hạn còn lại. Không báo metric chưa thực sự đo.
