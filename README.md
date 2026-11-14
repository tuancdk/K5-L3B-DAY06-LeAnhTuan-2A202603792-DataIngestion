# Day 06 · Data Ingestion Engineering

> ## Bài làm — Le Anh Tuan (2A202603792, L3B)
>
> **Corpus:** BCTC hợp nhất Q3/2026 của MB (PDF scan, 59 trang nội dung) · **OCR:** Google Cloud Vision · **Embedding:** OpenAI `text-embedding-3-small` (1536d) · **Store:** PostgreSQL 16 + pgvector 0.8.7, cosine exact · **Chiến lược chọn:** C (paragraph chunk + filter `section`)
>
> | Mục | File |
> |---|---|
> | Báo cáo cá nhân | [report/individual/2A202603792_LeAnhTuan.md](report/individual/2A202603792_LeAnhTuan.md) |
> | Manifest corpus (nguồn, quyền, trang, section) | [data/manifest.json](data/manifest.json) |
> | Benchmark 22 câu (16 dev / 6 test, FROZEN) | [benchmark/questions.json](benchmark/questions.json) |
> | Cấu hình và kết quả A/B/C | [artifacts/experiment_log.csv](artifacts/experiment_log.csv) |
> | Trace, chỉ số, log | `starter/results/` (`traces_*.jsonl`, `metrics_*.md`, `ingest_report.json`, `logs/`) |
> | Nhãn audit thủ công | `starter/results/audit/` (`audit_summary.md`) |
> | Kết quả OCR theo trang (tái lập không cần gọi API) | `starter/data/ocr/mbb_q3_2026/` |
>
> ### Cài đặt và chạy (Windows PowerShell, từ thư mục `starter/`)
>
> ```powershell
> python -m venv .venv; .venv\Scripts\Activate.ps1
> python -m pip install -r requirements.txt
> Copy-Item .env.example .env   # điền OPENAI_API_KEY, EMBEDDING_MODEL=text-embedding-3-small
>                               # (GOOGLE_VISION_API_KEY chỉ cần nếu OCR lại)
> docker compose -p day6_student_starter up -d --wait
> python scripts/check_setup.py
>
> # 1. (Tùy chọn) Tải PDF gốc và OCR lại — repo đã có sẵn kết quả OCR
> #    URL và sha256 trong data/manifest.json; lưu vào data/raw_mbb/
> python -m src.ocr data/raw_mbb/MBB_Baocaotaichinh_Q3_2026_Hopnhat.pdf data/ocr/mbb_q3_2026
> # 2. Kiểm tra corpus (>= 30 trang) và benchmark
> python scripts/build_manifest.py
> python scripts/validate_benchmark.py
> # 3. Chunk A/B, embed, lưu pgvector (nạp lại không nhân đôi)
> python -m src.run ingest
> # 4. Đánh giá A/B/C; test chỉ chạy khi benchmark FROZEN
> python -m src.run evaluate --split dev
> python -m src.run evaluate --split test
> # 5. Audit thủ công: tạo mẫu, tổng hợp nhãn
> python scripts/make_audit.py coherence; python scripts/make_audit.py test
> python scripts/score_audit.py
> # 6. Giao diện kiểm tra (chat truy xuất + trang kiểm chứng benchmark)
> streamlit run app.py
> ```
>
> Python 3.12.3, Docker 28.1.1; phiên bản thư viện ghi trong `starter/requirements.txt`. Không commit `.env` và PDF gốc.

---

**Lab cá nhân** · **4 giờ** · **Dữ liệu tự chọn**

Biến tài liệu thành kho tri thức có thể truy xuất cho AI agent, rồi đo xem cách chuẩn bị dữ liệu nào hiệu quả nhất.

| 📄 Dữ liệu | 🧩 Chiến lược | 🎯 Benchmark |
|---|---|---|
| Tối thiểu **30 trang nội dung** | Tối thiểu **3 cấu hình** | Tối thiểu **20 câu hỏi** |

## Bắt đầu nhanh

Dùng [starter PostgreSQL + pgvector](starter/README.md) để dựng database và hoàn thiện các hàm TODO, hoặc tự chọn vector store tương đương. Starter chỉ là khung: **bạn tự hiện thực pipeline và benchmark**.

## 1. Xây pipeline

![Đọc tài liệu, chuẩn hóa, chia chunk, tạo vector, lưu và kiểm tra](assets/ingestion-pipeline.svg)

Chọn tài liệu được phép sử dụng; giữ **ID tài liệu, nguồn, trang/vị trí** và metadata phù hợp. Dùng embedding model thật qua API hoặc chạy local; lưu và truy xuất bằng pgvector, Qdrant, Chroma hoặc một store tương đương.

## 2. So sánh và đánh giá

![Ba chiến lược được đánh giá trên cùng corpus và benchmark dev/test](assets/benchmark-workflow.svg)

Dùng cùng dữ liệu, model và câu hỏi cho cả ba cấu hình. Tách **15 câu dev / 5 câu test** trước khi chạy; chọn cấu hình trên dev, rồi đánh giá test sau khi cố định. Mỗi câu hỏi cần có đoạn bằng chứng và trang nguồn; thiết kế bộ câu hỏi và đánh giá theo [BENCHMARK.md](BENCHMARK.md).

Đánh giá theo năm khía cạnh: **độ chính xác truy xuất**, **tính mạch lạc của chunk**, **tính hữu dụng của metadata**, **chất lượng grounding** và **tác động của chiến lược dữ liệu**. Giải thích lựa chọn bằng kết quả, ví dụ lỗi và các đánh đổi.

## 3. Chuẩn bị bài nộp

1. Chạy pipeline và lưu cấu hình, kết quả top-3, trace cùng nhãn đánh giá thủ công.
2. Điền [thông tin học viên](STUDENT.md), [nhật ký thí nghiệm](artifacts/experiment_log.csv) và bản sao [mẫu báo cáo cá nhân](report/individual/MSSV_HoTen.md).
3. Kiểm tra mã nguồn, cách chạy, manifest dữ liệu và checklist trong [SUBMISSION.md](SUBMISSION.md).

## Tài liệu hướng dẫn

| Bạn cần tìm | Tài liệu |
|---|---|
| Setup baseline PostgreSQL + pgvector | [starter/README.md](starter/README.md) |
| Các bước làm và cách tính chỉ số | [GUIDE.md](GUIDE.md) |
| Tiến độ trong 4 giờ | [CHECKPOINTS.md](CHECKPOINTS.md) |
| Năm tiêu chí benchmark và bộ câu hỏi | [BENCHMARK.md](BENCHMARK.md) |
| Tiêu chí chấm điểm | [RUBRIC.md](RUBRIC.md) |
| Quy tắc sử dụng dữ liệu, AI và API | [RULES.md](RULES.md) |
| Cách nộp bài | [SUBMISSION.md](SUBMISSION.md) |

[data/format-example.json](data/format-example.json) chỉ minh họa định dạng, không phải corpus đủ điều kiện. Phần xử lý và benchmark do bạn tự hiện thực; không bắt buộc xây chatbot.
