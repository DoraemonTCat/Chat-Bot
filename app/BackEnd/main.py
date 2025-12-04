"""
Unified FastAPI Application
รวม RAG API และ LINE Webhook ไว้ใน app เดียว
"""
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from linebot import LineBotApi, WebhookParser
from linebot.models import TextSendMessage
from dotenv import load_dotenv
import os
import httpx
from typing import Optional
from pydantic import BaseModel

# Import StructuredRAG
from structured_rag import StructuredRAG


load_dotenv()

# ================== Configuration ==================
LINE_TOKEN = os.getenv("LINE_TOKEN")
LINE_SECRET = os.getenv("LINE_SECRET")
DIFY_API_KEY = os.getenv("DIFY_API_KEY")
DIFY_CHAT_URL = os.getenv("DIFY_CHAT_URL")
DIFY_DataSet_API_KEY = os.getenv("DIFY_DataSet_API_KEY")
DIFY_DATASET_ID = os.getenv("DIFY_DATASET_ID")
DIFY_API_URL = os.getenv("DIFY_API_URL", "http://localhost")

# LINE Bot
line_bot = LineBotApi(LINE_TOKEN)
parser = WebhookParser(LINE_SECRET)

# RAG Service
rag_service = StructuredRAG(
    dify_dataset_api_key=DIFY_DataSet_API_KEY,
    dify_url=DIFY_API_URL,
    dataset_id=DIFY_DATASET_ID
)

############ Config Models ############

class KnowledgeProcessRequest(BaseModel):
    dataset_id: Optional[str] = None
    auto_update: bool = True
    fetch_all: bool = True

class SearchRequest(BaseModel):
    query: str
    dataset_id: Optional[str] = None
    top_k: int = 5

class StructureRequest(BaseModel):
    dataset_id: Optional[str] = None

# ================== Main App ==================
app = FastAPI(
    title="Integrated RAG + LINE Bot API",
    description="รวม RAG API และ LINE Webhook ไว้ใน service เดียว",
    version="2.0.0"
)

# ================== Root Endpoints ==================
@app.get("/")
async def root():
    """Health check และแสดงข้อมูล services"""
    return {
        "status": "running",
        "version": "2.0.0",
        "services": {
            "rag_api": {
                "description": "Structured RAG API",
                "endpoints": [
                    "/api/datasets",
                    "/api/knowledge/process",
                    "/api/knowledge/search",
                    "/api/knowledge/structure",
                    "/api/status"
                ]
            },
            "line_webhook": {
                "description": "LINE Bot Webhook",
                "endpoint": "/webhook/line"
            }
        },
        "dataset_id": rag_service.dataset_id,
        "documents_loaded": len(rag_service.documents)
    }

# ================== RAG API Endpoints ==================
@app.get("/api/datasets")
async def list_datasets():
    """ดึงรายการ datasets ทั้งหมด"""
    try:
        datasets = await rag_service.list_all_datasets()
        return {
            "status": "success",
            "total": len(datasets),
            "datasets": [
                {
                    "id": ds.get("id"),
                    "name": ds.get("name"),
                    "description": ds.get("description"),
                    "created_at": ds.get("created_at")
                }
                for ds in datasets
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/knowledge/process")
async def process_knowledge(
    request: KnowledgeProcessRequest,
    background_tasks: BackgroundTasks
):
    """ประมวลผล Knowledge และสร้าง Structured RAG"""
    try:
        dataset_id = request.dataset_id or rag_service.dataset_id
        
        if not dataset_id:
            raise HTTPException(
                status_code=400,
                detail="Dataset ID is required"
            )
        
        if request.auto_update:
            background_tasks.add_task(
                rag_service.process_and_learn,
                dataset_id,
                request.fetch_all
            )
            return {
                "status": "processing",
                "message": "Knowledge processing started in background",
                "dataset_id": dataset_id
            }
        else:
            result = await rag_service.process_and_learn(
                dataset_id,
                request.fetch_all
            )
            return result
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/knowledge/search")
async def semantic_search(request: SearchRequest):
    """ค้นหาความรู้ด้วย Semantic Search"""
    try:
        dataset_id = request.dataset_id or rag_service.dataset_id
        
        if not dataset_id:
            raise HTTPException(
                status_code=400,
                detail="Dataset ID is required"
            )
        
        # ใช้ documents ที่โหลดไว้แล้ว ถ้าไม่มีค่อยดึงใหม่
        if not rag_service.documents:
            documents = await rag_service.fetch_knowledge_from_dify(dataset_id)
            if documents:
                rag_service.documents = rag_service.create_embeddings(documents)
        
        if not rag_service.documents:
            return {
                "query": request.query,
                "results": [],
                "message": "No documents found in dataset"
            }
        
        # ค้นหา
        results = rag_service.semantic_search(
            query=request.query,
            documents=rag_service.documents,
            top_k=request.top_k
        )
        
        return {
            "query": request.query,
            "dataset_id": dataset_id,
            "total_results": len(results),
            "results": [
                {
                    "content": r['document'].content,
                    "similarity": r['similarity'],
                    "metadata": r['document'].metadata
                }
                for r in results
            ]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/knowledge/structure")
async def get_structured_knowledge(request: StructureRequest):
    """ดึงโครงสร้างความรู้"""
    try:
        dataset_id = request.dataset_id or rag_service.dataset_id
        
        if not dataset_id:
            raise HTTPException(
                status_code=400,
                detail="Dataset ID is required"
            )
        
        # ใช้ documents ที่โหลดไว้แล้ว
        if not rag_service.documents:
            documents = await rag_service.fetch_knowledge_from_dify(dataset_id)
            if documents:
                rag_service.documents = rag_service.create_embeddings(documents)
        
        if not rag_service.documents:
            return {
                "dataset_id": dataset_id,
                "structure": {},
                "message": "No documents found"
            }
        
        structured = rag_service.structure_knowledge(rag_service.documents)
        
        return {
            "dataset_id": dataset_id,
            "structure": structured,
            "total_documents": len(rag_service.documents),
            "categories": list(structured['categories'].keys())
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/status")
async def get_status():
    """ตรวจสอบสถานะ RAG Service"""
    return {
        "status": "running",
        "dataset_id": rag_service.dataset_id,
        "documents_loaded": len(rag_service.documents),
        "embedding_model": "paraphrase-multilingual-MiniLM-L12-v2"
    }

# ================== LINE Webhook ==================
@app.post("/webhook/line")
async def line_webhook(request: Request):
    """
    LINE Webhook with RAG Integration
    ใช้ RAG เพื่อค้นหาความรู้ที่เกี่ยวข้องก่อนส่งไปยัง Dify
    """
    try:
        body = await request.body()
        signature = request.headers.get('X-Line-Signature', '')
        
        # Parse events
        try:
            events = parser.parse(body.decode('utf-8'), signature)
        except Exception as e:
            print(f"Invalid signature: {e}")
            raise HTTPException(status_code=400, detail="Invalid signature")
        
        for event in events:
            if event.type == 'message' and event.message.type == 'text':
                user_message = event.message.text
                user_id = event.source.user_id
                
                print(f"\n📩 Received message from {user_id}: {user_message}")
                
                # ถ้ายังไม่มี documents โหลดมา
                if not rag_service.documents:
                    print("📥 Loading documents for the first time...")
                    documents = await rag_service.fetch_knowledge_from_dify()
                    if documents:
                        rag_service.documents = rag_service.create_embeddings(documents)
                        print(f"✅ Loaded {len(rag_service.documents)} documents")
                
                # ค้นหาความรู้ที่เกี่ยวข้อง
                context = ""
                if rag_service.documents:
                    print("🔍 Searching for relevant knowledge...")
                    rag_results = rag_service.semantic_search(
                        query=user_message,
                        documents=rag_service.documents,
                        top_k=3
                    )
                    
                    if rag_results:
                        context = "\n\n".join([
                            f"📚 ความรู้ที่เกี่ยวข้อง {i+1} (คะแนน: {r['similarity']:.4f}):\n{r['document'].content[:500]}"
                            for i, r in enumerate(rag_results)
                        ])
                        print(f"✅ Found {len(rag_results)} relevant documents")
                
                # สร้าง payload สำหรับ Dify
                dify_payload = {
                    "query": f"{context}\n\n---\n\nคำถามจากผู้ใช้: {user_message}" if context else user_message,
                    "inputs": {},
                    "response_mode": "blocking",
                    "user": str(user_id)
                }
                
                # ส่งไปยัง Dify Chat API
                print("🤖 Sending to Dify...")
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        DIFY_CHAT_URL,
                        headers={
                            "Authorization": f"Bearer {DIFY_API_KEY}",
                            "Content-Type": "application/json"
                        },
                        json=dify_payload,
                        timeout=30
                    )
                    
                    dify_response = response.json()
                    reply_text = dify_response.get("answer", "ขออภัย ไม่สามารถตอบได้ในขณะนี้")
                
                print(f"💬 Replying: {reply_text[:100]}...")
                
                # ตอบกลับไปยัง LINE
                line_bot.reply_message(
                    event.reply_token,
                    TextSendMessage(text=reply_text)
                )
        
        return {"status": "ok"}
        
    except Exception as e:
        print(f"❌ Error in LINE webhook: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ================== Startup Event ==================
@app.on_event("startup")
async def startup_event():
    """โหลด documents เมื่อ app เริ่มทำงาน"""
    print("\n" + "="*70)
    print("🚀 Starting Integrated RAG + LINE Bot API")
    print("="*70)
    print(f"Dataset ID: {DIFY_DATASET_ID}")
    print(f"Dify URL: {DIFY_API_URL}")
    print("\n📥 Pre-loading documents from Dify...")
    
    try:
        documents = await rag_service.fetch_knowledge_from_dify()
        if documents:
            rag_service.documents = rag_service.create_embeddings(documents)
            print(f"✅ Successfully loaded {len(rag_service.documents)} documents")
        else:
            print("⚠️  No documents found. Please upload documents to Dify dataset.")
    except Exception as e:
        print(f"⚠️  Failed to pre-load documents: {e}")
        print("   Documents will be loaded on first request.")
    
    print("\n🎉 API is ready!")
    print("="*70 + "\n")

# ================== Main ==================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )