# Ngày 1 — Bài Tập & Phản Ánh
## Nền Tảng LLM API | Phiếu Thực Hành

**Thời lượng:** 1:30 giờ  
**Cấu trúc:** Lập trình cốt lõi (60 phút) → Bài tập mở rộng (30 phút)

---

## Phần 1 — Lập Trình Cốt Lõi (0:00–1:00)

Chạy các ví dụ trong Google Colab tại: https://colab.research.google.com/drive/172zCiXpLr1FEXMRCAbmZoqTrKiSkUERm?usp=sharing

Triển khai tất cả TODO trong `template.py`. Chạy `pytest tests/` để kiểm tra tiến độ.

**Điểm kiểm tra:** Sau khi hoàn thành 4 nhiệm vụ, chạy:
```bash
python template.py
```
Bạn sẽ thấy output so sánh phản hồi của GPT-4o và GPT-4o-mini.

---

## Phần 2 — Bài Tập Mở Rộng (1:00–1:30)

### Bài tập 2.1 — Độ Nhạy Của Temperature
Gọi `call_openai` với các giá trị temperature 0.0, 0.5, 1.0 và 1.5 sử dụng prompt **"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
> *Temperature càng thấp thì câu trả lời càng ổn định, ngắn gọn và ít biến đổi. Temperature càng cao thì phản hồi sáng tạo hơn, đa dạng hơn nhưng cũng dễ lan man hoặc kém chính xác hơn.*

**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> *Tôi sẽ đặt khoảng 0.0–0.3 vì chatbot hỗ trợ khách hàng cần trả lời nhất quán, chính xác và ít “sáng tạo” ngoài ý muốn.n*

---

### Bài tập 2.2 — Đánh Đổi Chi Phí
Xem xét kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người thực hiện 3 lần gọi API, mỗi lần trung bình ~350 token.

**Ước tính xem GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này:**
> *Với cùng số token, GPT-4o thường đắt hơn GPT-4o-mini khoảng 16–17 lần với input và khoảng 16–25 lần với output, tùy bảng giá đang dùng. Tổng workload là 10.000 × 3 × 350 = 10.500.000 token/ngày, nên chênh lệch chi phí là rất lớn khi scale. GPT-4o-mini từng được công bố giá 0.15 USD/1M input token và 0.60 USD/1M output token, thấp hơn rất nhiều so với GPT-4o.*

**Mô tả một trường hợp mà chi phí cao hơn của GPT-4o là xứng đáng, và một trường hợp GPT-4o-mini là lựa chọn tốt hơn:**
> *GPT-4o xứng đáng khi cần xử lý tác vụ khó như phân tích tài liệu phức tạp, suy luận nhiều bước, hoặc yêu cầu độ chính xác cao. GPT-4o-mini phù hợp hơn cho chatbot FAQ, tóm tắt ngắn, phân loại nội dung, hoặc các tác vụ số lượng lớn cần tiết kiệm chi phí.*

---

### Bài tập 2.3 — Trải Nghiệm Người Dùng với Streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì non-streaming lại phù hợp hơn?** (1 đoạn văn)
> *Streaming quan trọng khi phản hồi dài hoặc người dùng cần cảm giác hệ thống đang trả lời ngay, ví dụ chatbot, trợ lý viết bài, giải thích code hoặc tạo nội dung dài. Non-streaming phù hợp hơn khi phản hồi ngắn, cần xử lý toàn bộ kết quả trước khi hiển thị, hoặc khi hệ thống cần validate, parse JSON, lưu log rồi mới trả kết quả cho người dùng.*


## Danh Sách Kiểm Tra Nộp Bài
- [ ] Tất cả tests pass: `pytest tests/ -v`
- [ ] `call_openai` đã triển khai và kiểm thử
- [ ] `call_openai_mini` đã triển khai và kiểm thử
- [ ] `compare_models` đã triển khai và kiểm thử
- [ ] `streaming_chatbot` đã triển khai và kiểm thử
- [ ] `retry_with_backoff` đã triển khai và kiểm thử
- [ ] `batch_compare` đã triển khai và kiểm thử
- [ ] `format_comparison_table` đã triển khai và kiểm thử
- [ ] `exercises.md` đã điền đầy đủ
- [ ] Sao chép bài làm vào folder `solution` và đặt tên theo quy định 
