from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
import uvicorn

# Import class pipeline dự đoán của bạn
from pipeline import SummarizationPipeline
from src.constants import *
from src.utils import read_yaml
from pathlib import Path

params = read_yaml(Path(PARAMS_FILE_PATH))

ml_models = {}

# 1. Cơ chế Lifespan: Load model khi bật ser    ver, giải phóng RAM khi tắt server
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Đang khởi động Server và tải Champion Model...")
    try:
        ml_models["summarizer"] = SummarizationPipeline(model_name=params.MODEL_NAME)
        print("Sẵn sàng nhận request!")
        yield
    except Exception as e:
        print(f"Lỗi khởi tạo model: {e}")
        yield
    finally:
        ml_models.clear()
        print("Đã tắt Server và giải phóng tài nguyên.")

# 2. Khởi tạo ứng dụng FastAPI
app = FastAPI(
    title="API Tóm Tắt Văn Bản Tiếng Việt",
    description="Sử dụng Champion Model từ MLflow Registry",
    lifespan=lifespan
)

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    html_content = """
    <!DOCTYPE html>
    <html lang="vi">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Demo Tóm Tắt Văn Bản</title>
        <style>
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; max-width: 800px; margin: 40px auto; padding: 20px; background-color: #f9fafb; color: #333; }
            h1 { color: #2563eb; text-align: center; }
            .container { background: white; padding: 30px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
            input[type="url"] { width: 100%; padding: 12px 15px; margin-top: 10px; border: 1px solid #cbd5e1; border-radius: 6px; font-size: 16px; box-sizing: border-box; }
            input[type="number"] { padding: 8px; border: 1px solid #cbd5e1; border-radius: 6px; width: 100px; }
            button { background-color: #2563eb; color: white; border: none; padding: 12px 24px; font-size: 16px; border-radius: 6px; cursor: pointer; margin-top: 15px; width: 100%; transition: background-color 0.3s; }
            button:hover { background-color: #1d4ed8; }
            button:disabled { background-color: #93c5fd; cursor: not-allowed; }
            .result-box { margin-top: 20px; padding: 20px; background-color: #f1f5f9; border-left: 4px solid #2563eb; border-radius: 4px; min-height: 80px; white-space: pre-wrap; line-height: 1.6; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚀 Demo AI: Tóm Tắt Tiếng Việt</h1>
            
            <label for="maxLength"><b>Độ dài tối đa (Max Length):</b></label>
            <input type="number" id="maxLength" value="128" min="10" max="500">
            <br>
            
            <input type="url" id="inputText" placeholder="Dán link bài báo vào đây (VD: https://vnexpress.net/...)">
            
            <button id="submitBtn" onclick="runInference()">⚡ Chạy Mô Hình</button>
            
            <h3>📊 Kết quả:</h3>
            <div id="result" class="result-box">Kết quả tóm tắt sẽ hiển thị ở đây...</div>
        </div>

        <script>
            async function runInference() {
                const text = document.getElementById('inputText').value.trim();
                const maxLength = document.getElementById('maxLength').value;
                const resultDiv = document.getElementById('result');
                const btn = document.getElementById('submitBtn');

                if (!text) {
                    alert("⚠️ Vui lòng nhập URL bài báo!");
                    return;
                }

                // Hiệu ứng loading
                btn.disabled = true;
                btn.innerText = "⏳ Đang xử lý...";
                resultDiv.innerText = "Đang kết nối tới mô hình...";

                try {
                    // Gọi API /predict bằng Fetch
                    const response = await fetch('/predict', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({
                            url: text,
                            max_length: parseInt(maxLength)
                        })
                    });

                    const data = await response.json();

                    if (response.ok) {
                        resultDiv.innerText = data.summary;
                    } else {
                        resultDiv.innerText = "❌ Lỗi: " + data.detail;
                    }
                } catch (error) {
                    resultDiv.innerText = "❌ Lỗi kết nối tới máy chủ. Vui lòng thử lại.";
                    console.error("Error:", error);
                } finally {
                    // Reset nút
                    btn.disabled = false;
                    btn.innerText = "⚡ Chạy Mô Hình";
                }
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content, status_code=200)

class SummaryRequest(BaseModel):
    url: str
    max_length: int = 128

class SummaryResponse(BaseModel):
    summary: str

@app.post("/predict", response_model=SummaryResponse)
async def predict_summary(request: SummaryRequest):
    if not request.url or len(request.url.strip()) == 0:
        raise HTTPException(status_code=400, detail="Văn bản đầu vào không được để trống")
    
    try:
        # Lấy model đã load sẵn từ bộ nhớ và chạy dự đoán
        model_pipeline = ml_models["summarizer"]
        result = model_pipeline.predict(url=request.url, max_length=request.max_length)
        
        return SummaryResponse(summary=result)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi trong quá trình xử lý: {str(e)}")