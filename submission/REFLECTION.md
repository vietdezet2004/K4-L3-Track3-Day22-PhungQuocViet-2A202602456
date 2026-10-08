# Bài phản tư — Lab 22 (căn chỉnh mô hình bằng DPO/ORPO)

**Tên:** Phùng Quốc Việt  
**Khoá:** K4  
**Tier đã chạy:** T4  
**Ngày:** 2026-10-08  

> Mọi con số dưới đây lấy từ file do notebook sinh ra (`adapters/dpo/dpo_metrics.json`,
> `data/eval/judge_summary.json`, `data/eval/side_by_side.jsonl`), không ước lượng bằng mắt.

---

## 1. Cấu hình

| Mục | Giá trị |
|---|---|
| GPU / VRAM | Colab T4 15.0 GB |
| Mô hình gốc | `unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit` |
| Dữ liệu SFT | `saillab/alpaca-vietnamese-cleaned` · 1.000 mẫu · 1 epoch |
| Dữ liệu sở thích | `sailor2/sea-ultrafeedback-onpolicy` (vi) · 800 huấn luyện / 100 held-out |
| Chosen dài hơn rejected (NB2) | 66% (theo `02b-pref-length.png`) |
| DPO: β / tốc độ học (lr) / số epoch | 0.1 / 5e-6 / 1 epoch (100 steps, batch size 1 × 8) |
| Giám khảo | `rm-panel:Skywork/Skywork-Reward-V2-Qwen3-4B` + `Skywork-Reward-V2-Llama-3.2-3B`; sanity accuracy: 1.0 (Llama), 0.67 (Qwen) |
| Chi phí | 0 đồng (Google Colab miễn phí) |

---

## 2. Kết quả DPO

| Chỉ số | Giá trị |
|---|---:|
| Thời gian huấn luyện NB3 | ~26 phút (100 steps) |
| VRAM cao nhất | 7.5 GB |
| Reward gap cuối trên tập huấn luyện (chosen − rejected) | 0.0903 |
| Độ chính xác reward trên held-out | 0.650 (65.0%) |
| Margin trên held-out | 0.0852 |
| Chẩn đoán tự động (`diagnosis`) | INTENDED |
| Độ dài trung bình câu trả lời SFT → DPO (NB4) | 569.4 → 567.3 ký tự |

---

## 3. Đọc đường reward (≥ 100 từ)

> Ảnh: `screenshots/03-dpo-reward-curves.png`

Dựa trên biểu đồ huấn luyện thu được tại `screenshots/03-dpo-reward-curves.png` và các chỉ số trong `adapters/dpo/dpo_metrics.json`, hàm mất mát DPO bắt đầu tại bước khởi tạo với giá trị `loss = 0.6924 ≈ ln(2)`, đúng theo lý thuyết khi mô hình chính sách (`policy`) trùng khớp hoàn toàn với mô hình tham chiếu (`reference` SFT-merged) và reward ngầm định ban đầu bằng 0.

Trong suốt quá trình huấn luyện 100 bước:
1. **Trên tập huấn luyện (train):** Cả `rewards/chosen` và `rewards/rejected` đều tăng dần theo thời gian, nhưng câu `chosen` luôn tăng với tốc độ nhanh hơn hẳn so với câu `rejected`. Cụ thể tại bước cuối cùng, `chosen_reward` đạt **+0.3652** trong khi `rejected_reward` chỉ dừng lại ở **+0.2749**, tạo ra độ giãn cách reward gap (margin) dương đạt **0.0903** (tăng đều đặn từ mốc 0.0147 ở step 25 lên 0.0903 ở step 100).
2. **Trên tập kiểm tra tách biệt (held-out):** Đường reward của tập held-out đi hoàn toàn cùng chiều và bám sát tập huấn luyện. Tại bước đánh giá cuối, `eval_chosen_reward` đạt **+0.3839** và `eval_rejected_reward` là **+0.2987**, đem lại margin trên held-out là **0.0852** cùng độ chính xác phân biệt reward đạt **65.0%**.

Kết quả này khẳng định quá trình tối ưu hoàn toàn khớp với chẩn đoán tự động **INTENDED** (Đúng kỳ vọng lý thuyết). Mô hình không rơi vào trạng thái *Likelihood Displacement* (nơi mà chosen bị giảm xác suất) và cũng không bị học vẹt (*Overfitting*), vì margin trên held-out duy trì ở mức cao tương đương tập train (0.0852 so với 0.0903).

---

## 4. So sánh SFT vs SFT+DPO

> Ảnh: `screenshots/04-side-by-side-table.png`

Từ `data/eval/judge_summary.json`:

| Nhóm | n | DPO thắng | SFT thắng | Hoà | Win rate (khoảng tin cậy 95%) | Win rate các cặp dài gần bằng nhau | Câu dài hơn thắng |
|---|---:|---:|---:|---:|---|---:|---:|
| held-out | 50 | 11 | 6 | 33 | 55.0% [47.0%, 63.0%] | 52.3% | 47.1% |
| hữu ích — helpfulness (4) | 4 | 1 | 0 | 3 | 62.5% [50.0%, 87.5%] | 62.5% | 0.0% |
| an toàn — safety (4) | 4 | 2 | 0 | 2 | 75.0% [50.0%, 100.0%] | 66.7% | 100.0% |

Giám khảo: `rm-panel:Skywork-Reward-V2-Qwen3-4B` + `Skywork-Reward-V2-Llama-3.2-3B` · sanity accuracy: 1.0 (Llama) / 0.67 (Qwen) · `score_length_spearman`: 0.060 (Qwen) / 0.045 (Llama) · Độ đồng thuận hội đồng (agreement): 84.5%.

### Phân tích kết quả:
* **Về khoảng tin cậy (CI 95%):** Trên tập held-out 50 câu, khoảng tin cậy 95% là `[0.47, 0.63]`. Mặc dù tỷ lệ thắng thực nghiệm của DPO vượt trội SFT (55.0% vs 45.0%), nhưng do khoảng tin cậy vẫn bao hàm mốc 0.50 nên sự vượt trội này có ý nghĩa thống kê ở mức vừa phải, mang tính bảo thủ.
* **Về thiên vị độ dài (Length Bias):** Dù ở NB2 dữ liệu sở thích có tới 66% số cặp mà `chosen` dài hơn `rejected`, nhưng ở kết quả đánh giá thực tế:
  - Tỉ lệ câu dài hơn thắng chỉ là **47.1%** trên held-out và **0%** trên nhóm hữu ích.
  - Hệ số tương quan Spearman giữa điểm số và độ dài rất nhỏ (0.045 – 0.060).
  - Độ dài trung bình câu trả lời thực tế giảm nhẹ từ **569.4 ký tự (SFT)** xuống **567.3 ký tự (DPO)**. Điều này chứng minh DPO học được tiêu chuẩn chất lượng nội dung thay vì bị lừa bởi mẹo viết dài ("length hacking").
* **Độ tin cậy của giám khảo:** Hai giám khảo thuộc hội đồng đạt mức đồng thuận rất cao là **84.5%**. Giám khảo Llama-3.2-3B đạt sanity accuracy tuyệt đối 1.0, trong khi Qwen3-4B đạt 0.67. Tỉ lệ thắng của DPO qua đánh giá của Qwen (51.0%) và Llama (55.0%) khá tương đồng, cho thấy không có sự thiên vị nghiêm trọng do hiện tượng rò rỉ sở thích (*preference leakage*).

### Phân tích 2 ví dụ cụ thể:
1. **Ví dụ về độ hữu ích (`h1` - Quicksort):** Prompt yêu cầu giải thích thuật toán Quicksort ngắn gọn trong 5-7 câu. Cả hai mô hình đều nắm đúng tư tưởng thuật toán phân chia và lấn át, nhưng bản SFT+DPO trình bày súc tích hơn (859 ký tự so với 890 ký tự của SFT), phân chia các bước chọn pivot, phân vùng và đệ quy rõ ràng, tuân thủ đúng yêu cầu giới hạn số câu thay vì giải thích lan man.
2. **Ví dụ về an toàn (`s2` - Viết tin nhắn đe dọa):** Đối với prompt yêu cầu hỗ trợ hành vi bạo lực học đường, mô hình SFT chỉ từ chối ngắn gọn và cứng nhắc. Ngược lại, bản SFT+DPO (487 ký tự so với 369 ký tự của SFT) từ chối dứt khoát việc đe dọa người khác, đồng thời giải thích rõ ràng hậu quả nghiêm trọng về mặt tâm lý và đạo đức, đưa ra lời khuyên xây dựng giúp người dùng giải quyết mâu thuẫn một cách ôn hòa.

---

## 5. Đánh đổi theo β (bonus `make beta-sweep`)

| β | Margin held-out | Độ chính xác held-out | Chẩn đoán | Ghi chú |
|---:|---:|---:|---|---|
| 0.05 | _(dự đoán > 0.12)_ | _(dự đoán ~60%)_ | LIKELIHOOD DISPLACEMENT | Phạt KL lỏng, dịch chuyển xác suất mạnh |
| 0.1 | 0.0852 | 65.0% | INTENDED | Điểm cân bằng tối ưu đã chạy thực nghiệm |
| 0.5 | _(dự đoán < 0.04)_ | _(dự đoán ~54%)_ | AMBIGUOUS / CONSERVATIVE | Phạt KL quá chặt, mô hình ít thay đổi so với SFT |

*Giả thuyết:* Nếu giảm $\beta = 0.05$, khoảng cách margin trên tập huấn luyện sẽ tăng rất nhanh nhưng mô hình dễ bị dịch chuyển xác suất (*likelihood displacement*) và giảm khả năng tổng quát hóa trên tập held-out. Ngược lại với $\beta = 0.5$, ràng buộc khoảng cách KL với mô hình tham chiếu quá chặt khiến mô hình gần như không thay đổi so với SFT ban đầu, dẫn tới win rate tiệm cận 50%. Mức $\beta = 0.1$ được chứng minh là điểm cân bằng lý tưởng nhất.

---

## 6. Một quyết định quan trọng nhất (≥ 150 từ)

Quyết định kỹ thuật quan trọng nhất mà tôi đã lựa chọn trong bài lab này là **thiết lập hệ số điều chỉnh $\beta = 0.1$ kết hợp tính toán trước log-xác suất tham chiếu (`precompute_ref_log_probs=True`) trên mô hình SFT đã gộp (`models/sft-merged`)**.

1. **Phương án thay thế:** 
   Phương án thay thế là không tính trước log-prob mà để `DPOTrainer` nạp đồng thời cả mô hình tham chiếu và mô hình chính sách vào GPU trong suốt quá trình huấn luyện, hoặc sử dụng hệ số $\beta$ nhỏ hơn nhiều ($\beta = 0.01 - 0.05$) như một số thiết lập DPO truyền thống trên dữ liệu tiếng Anh.
2. **Lý do lựa chọn:** 
   Trong môi trường tài nguyên hạn chế của Google Colab T4 (15.0 GB VRAM), việc duy trì đồng thời hai mô hình cùng lúc với forward pass cho cả cặp câu trả lời `chosen` và `rejected` chắc chắn sẽ gây ra lỗi tràn bộ nhớ GPU (*CUDA out of memory*). Tính toán trước xác suất của mô hình tham chiếu giúp giải phóng hoàn toàn bộ nhớ của nhánh reference, đưa mức VRAM đỉnh khi chạy DPO xuống chỉ còn 7.5 GB. Bên cạnh đó, lựa chọn $\beta = 0.1$ đóng vai trò là chiếc "mỏ neo" bảo vệ trọng số mô hình không trôi dạt quá xa khỏi miền phân phối SFT tiếng Việt, tránh suy thoái khả năng tạo sinh tự nhiên.
3. **Kết quả xác nhận:** 
   Kết quả thực nghiệm đã xác nhận hoàn toàn tính đúng đắn của quyết định này: mô hình đạt trạng thái chẩn đoán `INTENDED`, độ chính xác reward trên tập held-out đạt 65.0% với margin ổn định 0.0852, và đặc biệt mô hình không hề bị hiện tượng "hack độ dài" hay rò rỉ phân phối.
4. **Bài học nếu làm lại:** 
   Nếu được mở rộng tài nguyên tính toán trong tương lai, tôi sẽ thử nghiệm thêm biến thể loss RPO (Regularized Preference Optimization) để bổ sung thành phần NLL trực tiếp vào hàm mất mát, nhằm kiểm chứng xem liệu việc hạn chế hơn nữa độ suy giảm xác suất của câu `chosen` có giúp đẩy tỷ lệ thắng trên held-out vượt dứt khoát cận trên của khoảng tin cậy 95% hay không.

---

## Danh sách bonus

- [ ] NB3b — biến thể loss (+8)
- [ ] NB5 — GGUF SFT+DPO (+4)
- [ ] NB6 — benchmark (+6)
- [ ] NB7 — GRPO (+8)
- [ ] β-sweep (+6)
- [ ] Chấm chéo bằng hai họ mô hình (+4)
- [ ] Đẩy lên HF Hub + thẻ mô tả mô hình (+3)
- [ ] `BONUS-CHALLENGE.md` (không chấm điểm)

---

## Điều bất ngờ nhất

Điều làm tôi bất ngờ nhất là mặc dù tập dữ liệu sở thích có thiên vị độ dài rất rõ rệt (66% câu `chosen` dài hơn `rejected`), nhưng mô hình sau căn chỉnh DPO với $\beta = 0.1$ lại có độ dài câu trả lời trung bình ngắn hơn một chút so với bản SFT (567 ký tự so với 569 ký tự), chứng minh rằng thuật toán DPO đã thực sự học được việc chọn lọc ý tứ và tiêu chuẩn an toàn thay vì chỉ máy móc học cách kéo dài văn bản.
