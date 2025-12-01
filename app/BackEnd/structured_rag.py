import os
from typing import List, Dict, Any
from datetime import datetime
import httpx
from sentence_transformers import SentenceTransformer
import numpy as np
from pydantic import BaseModel

class Document(BaseModel):
    """โครงสร้างข้อมูลเอกสาร"""
    id: str
    content: str
    metadata: Dict[str, Any]
    embedding: List[float] = []

class StructuredRAG:
    """
    Structured RAG Service สำหรับ Dify Knowledge Base
    """
    
    def __init__(self, dify_api_key: str, dify_url: str = "http://localhost"):
        self.dify_api_key = dify_api_key
        self.dify_url = dify_url
        self.embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        self.documents = []
        
    async def fetch_knowledge_from_dify(self, dataset_id: str) -> List[Document]:
        """
        ดึง Knowledge จาก Dify Dataset
        """
        async with httpx.AsyncClient() as client:
            headers = {
                "Authorization": f"Bearer {self.dify_api_key}",
                "Content-Type": "application/json"
            }
            
            # API endpoint สำหรับดึง documents
            url = f"{self.dify_url}/v1/datasets/{dataset_id}/documents"
            
            try:
                response = await client.get(url, headers=headers)
                response.raise_for_status()
                data = response.json()
                
                documents = []
                for doc in data.get('data', []):
                    documents.append(Document(
                        id=doc['id'],
                        content=doc['content'],
                        metadata={
                            'name': doc.get('name', ''),
                            'created_at': doc.get('created_at', ''),
                            'word_count': doc.get('word_count', 0)
                        }
                    ))
                
                return documents
                
            except Exception as e:
                print(f"Error fetching knowledge: {e}")
                return []
    
    def create_embeddings(self, documents: List[Document]) -> List[Document]:
        """
        สร้าง embeddings สำหรับเอกสาร
        """
        texts = [doc.content for doc in documents]
        embeddings = self.embedding_model.encode(texts)
        
        for doc, emb in zip(documents, embeddings):
            doc.embedding = emb.tolist()
        
        return documents
    
    def structure_knowledge(self, documents: List[Document]) -> Dict[str, Any]:
        """
        จัดโครงสร้างความรู้แบบ Hierarchical
        """
        structured = {
            'metadata': {
                'total_documents': len(documents),
                'created_at': datetime.now().isoformat(),
                'embedding_model': 'paraphrase-multilingual-MiniLM-L12-v2'
            },
            'categories': {},
            'documents': []
        }
        
        # จัดหมวดหมู่ตาม metadata
        for doc in documents:
            category = doc.metadata.get('name', 'uncategorized')
            if category not in structured['categories']:
                structured['categories'][category] = []
            
            structured['categories'][category].append({
                'id': doc.id,
                'content': doc.content,
                'embedding': doc.embedding,
                'metadata': doc.metadata
            })
            
            structured['documents'].append({
                'id': doc.id,
                'content': doc.content,
                'embedding': doc.embedding,
                'category': category,
                'metadata': doc.metadata
            })
        
        return structured
    
    def semantic_search(self, query: str, documents: List[Document], top_k: int = 5) -> List[Dict]:
        """
        ค้นหาเอกสารที่เกี่ยวข้องด้วย Semantic Search
        """
        query_embedding = self.embedding_model.encode([query])[0]
        
        # คำนวณ cosine similarity
        similarities = []
        for doc in documents:
            doc_emb = np.array(doc.embedding)
            similarity = np.dot(query_embedding, doc_emb) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc_emb)
            )
            similarities.append({
                'document': doc,
                'similarity': float(similarity)
            })
        
        # เรียงลำดับตาม similarity
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        return similarities[:top_k]
    
    async def update_dify_knowledge(
        self, 
        dataset_id: str, 
        structured_data: Dict[str, Any]
    ) -> bool:
        """
        อัพเดท Knowledge กลับไปที่ Dify พร้อม Structured Data
        """
        async with httpx.AsyncClient() as client:
            headers = {
                "Authorization": f"Bearer {self.dify_api_key}",
                "Content-Type": "application/json"
            }
            
            url = f"{self.dify_url}/v1/datasets/{dataset_id}/documents"
            
            # สร้างเอกสารใหม่จาก structured data
            for category, docs in structured_data['categories'].items():
                for doc in docs:
                    payload = {
                        "name": f"structured_{category}_{doc['id']}",
                        "text": doc['content'],
                        "indexing_technique": "high_quality",
                        "process_rule": {
                            "mode": "custom",
                            "rules": {
                                "pre_processing_rules": [
                                    {"id": "remove_extra_spaces", "enabled": True},
                                    {"id": "remove_urls_emails", "enabled": True}
                                ],
                                "segmentation": {
                                    "separator": "\\n",
                                    "max_tokens": 500
                                }
                            }
                        }
                    }
                    
                    try:
                        response = await client.post(url, headers=headers, json=payload)
                        response.raise_for_status()
                    except Exception as e:
                        print(f"Error updating document {doc['id']}: {e}")
                        return False
            
            return True
    
    async def process_and_learn(self, dataset_id: str) -> Dict[str, Any]:
        """
        กระบวนการหลักของ Structured RAG
        1. ดึง Knowledge จาก Dify
        2. สร้าง Embeddings
        3. จัดโครงสร้าง
        4. อัพเดทกลับไปที่ Dify
        """
        # 1. ดึง Knowledge
        print("Fetching knowledge from Dify...")
        documents = await self.fetch_knowledge_from_dify(dataset_id)
        
        if not documents:
            return {"status": "error", "message": "No documents found"}
        
        # 2. สร้าง Embeddings
        print("Creating embeddings...")
        documents = self.create_embeddings(documents)
        
        # 3. จัดโครงสร้าง
        print("Structuring knowledge...")
        structured_data = self.structure_knowledge(documents)
        
        # 4. อัพเดทกลับไปที่ Dify
        print("Updating Dify knowledge...")
        success = await self.update_dify_knowledge(dataset_id, structured_data)
        
        return {
            "status": "success" if success else "partial_success",
            "total_documents": len(documents),
            "categories": list(structured_data['categories'].keys()),
            "timestamp": datetime.now().isoformat()
        }
        
      