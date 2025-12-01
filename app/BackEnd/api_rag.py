from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os
from dotenv import load_dotenv

# Import StructuredRAG จากไฟล์ก่อนหน้า
# from structured_rag import StructuredRAG

load_dotenv()

app = FastAPI(title="Structured RAG API for Dify")

# Initialize RAG Service
rag_service = StructuredRAG(
    dify_api_key=os.getenv("DIFY_API_KEY"),
    dify_url=os.getenv("DIFY_CHAT_URL", "http://localhost")
)

class KnowledgeProcessRequest(BaseModel):
    dataset_id: str
    auto_update: bool = True

class SearchRequest(BaseModel):
    query: str
    dataset_id: str
    top_k: int = 5

class UpdateKnowledgeRequest(BaseModel):
    dataset_id: str
    documents: List[Dict[str, Any]]

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "running",
        "service": "Structured RAG for Dify",
        "version": "1.0.0"
    }

@app.post("/api/knowledge/process")
async def process_knowledge(
    request: KnowledgeProcessRequest,
    background_tasks: BackgroundTasks
):
    """
    ประมวลผล Knowledge จาก Dify และสร้าง Structured RAG
    
    Process:
    1. ดึง Knowledge จาก Dify Dataset
    2. สร้าง Embeddings
    3. จัดโครงสร้างความรู้
    4. อัพเดทกลับไปที่ Dify (ถ้า auto_update=True)
    """
    try:
        if request.auto_update:
            # Run in background
            background_tasks.add_task(
                rag_service.process_and_learn,
                request.dataset_id
            )
            return {
                "status": "processing",
                "message": "Knowledge processing started in background",
                "dataset_id": request.dataset_id
            }
        else:
            # Run synchronously
            result = await rag_service.process_and_learn(request.dataset_id)
            return result
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/knowledge/search")
async def semantic_search(request: SearchRequest):
    """
    ค้นหาความรู้ที่เกี่ยวข้องด้วย Semantic Search
    """
    try:
        # ดึง documents
        documents = await rag_service.fetch_knowledge_from_dify(request.dataset_id)
        
        if not documents:
            return {"results": [], "message": "No documents found"}
        
        # สร้าง embeddings
        documents = rag_service.create_embeddings(documents)
        
        # ค้นหา
        results = rag_service.semantic_search(
            query=request.query,
            documents=documents,
            top_k=request.top_k
        )
        
        return {
            "query": request.query,
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
async def get_structured_knowledge(dataset_id: str):
    """
    ดึงโครงสร้างความรู้ที่จัดเรียบร้อยแล้ว
    """
    try:
        # ดึง documents
        documents = await rag_service.fetch_knowledge_from_dify(dataset_id)
        
        if not documents:
            return {"structure": {}, "message": "No documents found"}
        
        # สร้าง embeddings
        documents = rag_service.create_embeddings(documents)
        
        # จัดโครงสร้าง
        structured = rag_service.structure_knowledge(documents)
        
        return {
            "dataset_id": dataset_id,
            "structure": structured
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/knowledge/update")
async def update_knowledge(request: UpdateKnowledgeRequest):
    """
    อัพเดท Knowledge กลับไปที่ Dify
    """
    try:
        success = await rag_service.update_dify_knowledge(
            dataset_id=request.dataset_id,
            structured_data={'categories': {'manual': request.documents}}
        )
        
        return {
            "status": "success" if success else "failed",
            "dataset_id": request.dataset_id,
            "documents_updated": len(request.documents)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/webhook/line")
async def line_webhook_with_rag(request: dict):
    """
    LINE Webhook ที่ผสม RAG
    """
    try:
        events = request.get('events', [])
        
        for event in events:
            if event.get('type') == 'message' and event['message']['type'] == 'text':
                user_message = event['message']['text']
                
                # ค้นหาความรู้ที่เกี่ยวข้อง
                search_request = SearchRequest(
                    query=user_message,
                    dataset_id=os.getenv("DIFY_DATASET_ID", ""),
                    top_k=3
                )
                
                rag_results = await semantic_search(search_request)
                
                # สร้าง context สำหรับ Dify
                context = "\n\n".join([
                    f"ความรู้ที่เกี่ยวข้อง {i+1}:\n{r['content']}"
                    for i, r in enumerate(rag_results['results'])
                ])
                
                # ส่งต่อไปยัง Dify พร้อม context
                dify_payload = {
                    "query": f"{context}\n\nคำถาม: {user_message}",
                    "inputs": {},
                    "response_mode": "blocking",
                    "user": str(event['source']['userId'])
                }
                
                # ส่งไปยัง Dify
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        os.getenv("DIFY_CHAT_URL"),
                        headers={
                            "Authorization": f"Bearer {os.getenv('DIFY_API_KEY')}",
                            "Content-Type": "application/json"
                        },
                        json=dify_payload
                    )
                    
                    dify_response = response.json()
                    reply_text = dify_response.get("answer", "ขออภัย ไม่สามารถตอบได้")
                
                # ส่งกลับไปยัง LINE
                # ... ใช้ line_bot.reply_message() ตามเดิม
                
        return {"status": "ok"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    
