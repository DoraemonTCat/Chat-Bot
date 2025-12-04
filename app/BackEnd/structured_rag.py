import os
from typing import List, Dict, Any, Optional
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
    ใช้ Dataset API Key สำหรับการเข้าถึง Dataset
    """
    
    def __init__(
        self, 
        dify_dataset_api_key: str,
        dify_url: str = "http://localhost",
        dataset_id: Optional[str] = None
    ):
        """
        Args:
            dify_dataset_api_key: Dataset API Key (dataset-xxx)
            dify_url: Dify Base URL
            dataset_id: Dataset ID (ถ้าไม่ระบุจะใช้จาก .env)
        """
        self.dify_dataset_api_key = dify_dataset_api_key
        self.dify_url = dify_url
        self.dataset_id = dataset_id or os.getenv("DIFY_DATASET_ID")
        self.embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        self.documents = []
        
        print(f"🔧 StructuredRAG initialized")
        print(f"   Dataset API Key: {self.dify_dataset_api_key[:20]}...")
        print(f"   Dataset ID: {self.dataset_id}")
        print(f"   Base URL: {self.dify_url}")
    
    async def list_all_datasets(self) -> List[Dict[str, Any]]:
        """
        ดึงรายการ datasets ทั้งหมด
        ใช้ Dataset API Key ในการเข้าถึง
        """
        async with httpx.AsyncClient() as client:
            headers = {
                "Authorization": f"Bearer {self.dify_dataset_api_key}",
                "Content-Type": "application/json"
            }
            
            # API endpoint สำหรับดึงรายการ datasets
            url = f"{self.dify_url}/v1/datasets"
            
            try:
                print(f"\n📋 Listing all datasets...")
                print(f"   URL: {url}")
                
                response = await client.get(url, headers=headers, timeout=30)
                response.raise_for_status()
                data = response.json()
                
                datasets = data.get('data', [])
                print(f"   ✅ Found {len(datasets)} datasets")
                
                for ds in datasets:
                    print(f"      - {ds.get('name')} (ID: {ds.get('id')})")
                
                return datasets
                
            except httpx.HTTPStatusError as e:
                print(f"   ❌ HTTP Error {e.response.status_code}: {e.response.text}")
                return []
            except Exception as e:
                print(f"   ❌ Error listing datasets: {e}")
                return []
    
    async def fetch_document_segments(
        self,
        dataset_id: str,
        document_id: str
    ) -> str:
        """
        ดึง segments (เนื้อหาเต็ม) ของ document
        """
        async with httpx.AsyncClient() as client:
            headers = {
                "Authorization": f"Bearer {self.dify_dataset_api_key}",
                "Content-Type": "application/json"
            }
            
            url = f"{self.dify_url}/v1/datasets/{dataset_id}/documents/{document_id}/segments"
            
            try:
                response = await client.get(url, headers=headers, timeout=30)
                response.raise_for_status()
                data = response.json()
                
                segments = data.get('data', [])
                
                # รวม content จากทุก segments
                content_parts = []
                for seg in segments:
                    seg_content = seg.get('content', '')
                    if seg_content:
                        content_parts.append(seg_content)
                
                full_content = "\n\n".join(content_parts)
                print(f"      ✅ Retrieved {len(segments)} segments ({len(full_content)} chars)")
                
                return full_content
                
            except Exception as e:
                print(f"      ⚠️  Failed to fetch segments: {e}")
                return ""
    
    async def fetch_knowledge_from_dify(
        self, 
        dataset_id: Optional[str] = None,
        page: int = 1,
        limit: int = 100
    ) -> List[Document]:
        """
        ดึง Knowledge/Documents จาก Dify Dataset
        ใช้ Dataset API Key และ Dataset ID
        
        Args:
            dataset_id: Dataset ID (ถ้าไม่ระบุจะใช้จาก __init__)
            page: หน้าที่ต้องการดึง (เริ่มจาก 1)
            limit: จำนวน documents ต่อหน้า
        """
        target_dataset_id = dataset_id or self.dataset_id
        
        if not target_dataset_id:
            print("❌ Dataset ID not provided!")
            return []
        
        async with httpx.AsyncClient() as client:
            headers = {
                "Authorization": f"Bearer {self.dify_dataset_api_key}",
                "Content-Type": "application/json"
            }
            
            # API endpoint สำหรับดึง documents
            url = f"{self.dify_url}/v1/datasets/{target_dataset_id}/documents"
            params = {
                "page": page,
                "limit": limit
            }
            
            try:
                print(f"\n📄 Fetching documents from dataset: {target_dataset_id}")
                print(f"   URL: {url}")
                print(f"   Params: page={page}, limit={limit}")
                
                response = await client.get(
                    url, 
                    headers=headers, 
                    params=params,
                    timeout=30
                )
                response.raise_for_status()
                data = response.json()
                
                documents = []
                doc_list = data.get('data', [])
                
                print(f"   ✅ Found {len(doc_list)} documents")
                
                for idx, doc in enumerate(doc_list, 1):
                    doc_id = doc.get('id')
                    doc_name = doc.get('name', 'Unknown')
                    word_count = doc.get('word_count', 0)
                    
                    # แสดงข้อมูลเอกสาร
                    print(f"\n   📝 Document {idx}:")
                    print(f"      ID: {doc_id}")
                    print(f"      Name: {doc_name}")
                    print(f"      Word Count: {word_count}")
                    print(f"      Created: {doc.get('created_at', 'N/A')}")
                    
                    # ดึงเนื้อหาจาก segments
                    print(f"      🔄 Fetching segments...")
                    doc_content = await self.fetch_document_segments(
                        target_dataset_id, 
                        doc_id
                    )
                    
                    # ถ้ายังไม่มี content ลองดูจาก response ตรง
                    if not doc_content:
                        doc_content = doc.get('content', '')
                        if doc_content:
                            print(f"      ✅ Using direct content ({len(doc_content)} chars)")
                    
                    # ถ้ายังไม่มีเลย แสดงว่าเป็น document ว่าง
                    if not doc_content:
                        print(f"      ⚠️  No content available for this document")
                    
                    documents.append(Document(
                        id=doc_id,
                        content=doc_content,
                        metadata={
                            'name': doc_name,
                            'created_at': doc.get('created_at', ''),
                            'updated_at': doc.get('updated_at', ''),
                            'word_count': word_count,
                            'character_count': doc.get('character_count', 0),
                            'indexing_status': doc.get('indexing_status', ''),
                            'dataset_id': target_dataset_id,
                            'has_content': bool(doc_content)
                        }
                    ))
                
                # ตรวจสอบว่ามีหน้าถัดไปหรือไม่
                total = data.get('total', 0)
                has_more = data.get('has_more', False)
                
                print(f"\n   📊 Summary:")
                print(f"      Total documents in dataset: {total}")
                print(f"      Retrieved this page: {len(doc_list)}")
                print(f"      Has more pages: {has_more}")
                
                return documents
                
            except httpx.HTTPStatusError as e:
                print(f"\n   ❌ HTTP Error {e.response.status_code}")
                print(f"      Response: {e.response.text}")
                return []
            except Exception as e:
                print(f"\n   ❌ Error fetching knowledge: {e}")
                return []
    
    async def fetch_all_documents(
        self, 
        dataset_id: Optional[str] = None
    ) -> List[Document]:
        """
        ดึง documents ทั้งหมดจาก dataset (รองรับ pagination)
        """
        all_documents = []
        page = 1
        limit = 100
        
        while True:
            documents = await self.fetch_knowledge_from_dify(
                dataset_id=dataset_id,
                page=page,
                limit=limit
            )
            
            if not documents:
                break
            
            all_documents.extend(documents)
            
            # ถ้าได้น้อยกว่า limit แสดงว่าหมดแล้ว
            if len(documents) < limit:
                break
            
            page += 1
        
        print(f"\n   ✅ Total documents fetched: {len(all_documents)}")
        return all_documents
    
    def create_embeddings(self, documents: List[Document]) -> List[Document]:
        """
        สร้าง embeddings สำหรับเอกสาร
        """
        if not documents:
            print("⚠️  No documents to create embeddings")
            return []
        
        print(f"\n🧠 Creating embeddings for {len(documents)} documents...")
        
        texts = [doc.content for doc in documents]
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        
        for doc, emb in zip(documents, embeddings):
            doc.embedding = emb.tolist()
        
        print(f"   ✅ Embeddings created successfully")
        return documents
    
    def structure_knowledge(self, documents: List[Document]) -> Dict[str, Any]:
        """
        จัดโครงสร้างความรู้แบบ Hierarchical
        """
        print(f"\n🏗️  Structuring knowledge...")
        
        structured = {
            'metadata': {
                'total_documents': len(documents),
                'created_at': datetime.now().isoformat(),
                'embedding_model': 'paraphrase-multilingual-MiniLM-L12-v2',
                'dataset_id': self.dataset_id
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
        
        print(f"   ✅ Knowledge structured into {len(structured['categories'])} categories")
        return structured
    
    def semantic_search(
        self, 
        query: str, 
        documents: List[Document], 
        top_k: int = 5
    ) -> List[Dict]:
        """
        ค้นหาเอกสารที่เกี่ยวข้องด้วย Semantic Search
        """
        print(f"\n🔍 Searching for: '{query}'")
        
        query_embedding = self.embedding_model.encode([query])[0]
        
        # คำนวณ cosine similarity
        similarities = []
        for doc in documents:
            if not doc.embedding:
                continue
                
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
        
        top_results = similarities[:top_k]
        
        print(f"   ✅ Found {len(top_results)} relevant documents")
        for idx, result in enumerate(top_results, 1):
            print(f"      {idx}. Similarity: {result['similarity']:.4f} - {result['document'].metadata.get('name', 'Unknown')}")
        
        return top_results
    
    async def process_and_learn(
        self, 
        dataset_id: Optional[str] = None,
        fetch_all: bool = True
    ) -> Dict[str, Any]:
        """
        กระบวนการหลักของ Structured RAG
        1. ดึง Knowledge จาก Dify
        2. สร้าง Embeddings
        3. จัดโครงสร้าง
        
        Args:
            dataset_id: Dataset ID (ถ้าไม่ระบุจะใช้จาก __init__)
            fetch_all: ดึง documents ทั้งหมดหรือไม่ (pagination)
        """
        print("\n" + "="*60)
        print("🚀 Starting Structured RAG Process")
        print("="*60)
        
        # 1. ดึง Knowledge
        if fetch_all:
            documents = await self.fetch_all_documents(dataset_id)
        else:
            documents = await self.fetch_knowledge_from_dify(dataset_id)
        
        if not documents:
            return {
                "status": "error", 
                "message": "No documents found in dataset"
            }
        
        # 2. สร้าง Embeddings
        documents = self.create_embeddings(documents)
        
        # 3. จัดโครงสร้าง
        structured_data = self.structure_knowledge(documents)
        
        # เก็บ documents ไว้ใช้งาน
        self.documents = documents
        
        print("\n" + "="*60)
        print("✅ Structured RAG Process Completed")
        print("="*60)
        
        return {
            "status": "success",
            "total_documents": len(documents),
            "categories": list(structured_data['categories'].keys()),
            "timestamp": datetime.now().isoformat(),
            "dataset_id": dataset_id or self.dataset_id
        }