# Deploy lên Vercel

Web UI so sánh **Chatbot (no tools)** vs **ReAct Agent (TVmaze tools)**.
Backend là một Flask serverless function (`api/index.py`), frontend là trang HTML
phục vụ ngay từ `/`.

## Kiến trúc

```
api/index.py     # Flask app (biến `app`) + giao diện HTML inline
                 #   GET  /            -> UI
                 #   GET  /api/health  -> provider nào đã cấu hình
                 #   POST /api/ask     -> chạy chatbot và/hoặc agent
vercel.json      # rewrite mọi route -> /api/index, maxDuration 60s, includeFiles src/**
requirements.txt # deps cho Vercel (KHÔNG có llama-cpp-python)
```

Vercel chỉ chạy được provider **OpenAI** và **Google Gemini** (provider `local`
cần file model trên đĩa nên đã bị chặn). `src/telemetry/logger.py` tự động bỏ ghi
file log khi filesystem read-only (như trên Vercel).

## Chạy thử ở máy

```bash
pip install -r requirements.txt
cp .env.example .env          # điền OPENAI_API_KEY hoặc GEMINI_API_KEY
python api/index.py           # mở http://localhost:5000
```

## Deploy

### Cách 1 — Git + Vercel Dashboard (khuyến nghị)

1. Push code lên GitHub/GitLab/Bitbucket.
2. Vào https://vercel.com → **Add New… → Project** → import repo này.
3. Framework Preset để **Other** (Vercel tự nhận `requirements.txt` + `api/`).
4. Mục **Environment Variables**, thêm:
   - `DEFAULT_PROVIDER` = `openai` (hoặc `google`)
   - **Nếu dùng OpenAI / endpoint OpenAI-compatible:**
     - `OPENAI_API_KEY` = `sk-...`
     - `OPENAI_BASE_URL` = (chỉ đặt khi dùng proxy như opencode zen, vd `https://opencode.ai/zen/go/v1`; bỏ trống nếu dùng OpenAI chính thức)
     - `OPENAI_MODEL` = `gpt-4o-mini` (tuỳ chọn; hoặc tên model của proxy, vd `deepseek-v4-flash`)
   - **Nếu dùng Gemini:**
     - `GEMINI_API_KEY` = `...`
     - `GEMINI_MODEL` = `gemini-2.0-flash` (tuỳ chọn)

   ⚠️ **Lưu ý:** `DEFAULT_MODEL` chỉ áp dụng cho OpenAI. Tên model **không dùng chung**
   giữa các provider — đặt `gpt-4o-mini` cho Gemini sẽ gây lỗi `404 model not found`.
   Hãy dùng `GEMINI_MODEL` / `OPENAI_MODEL` riêng cho từng provider.
5. Bấm **Deploy**. Xong sẽ có URL dạng `https://<project>.vercel.app`.

Mỗi lần push lên nhánh chính, Vercel tự build lại.

### Cách 2 — Vercel CLI

```bash
npm i -g vercel
vercel login

# thêm biến môi trường cho môi trường production
vercel env add OPENAI_API_KEY production
vercel env add DEFAULT_PROVIDER production    # nhập: openai
vercel env add DEFAULT_MODEL production       # nhập: gpt-4o-mini  (tuỳ chọn)

vercel --prod
```

Kiểm tra sau khi deploy: mở `https://<project>.vercel.app/api/health` để xem
provider đã được cấu hình hay chưa.

## Lưu ý / Giới hạn

- **Timeout:** `maxDuration` đặt 60s. Agent gọi LLM nhiều bước nên câu hỏi phức tạp
  có thể chậm; giảm `max_steps` hoặc dùng model nhanh (`gpt-4o-mini`,
  `gemini-1.5-flash`) nếu bị timeout. Plan Hobby tối đa 60s, Pro tối đa 300s.
- **Bí mật:** đừng commit `.env` (đã có trong `.vercelignore`). Khoá API chỉ đặt ở
  Environment Variables trên Vercel.
- **Local model:** không chạy được trên Vercel. Muốn dùng ở máy thì cài thêm
  `pip install -r requirements-local.txt` và đặt `DEFAULT_PROVIDER=local`.
