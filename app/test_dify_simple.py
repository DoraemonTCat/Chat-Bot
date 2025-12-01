import httpx
import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

async def test():
    # Config
    api_key = os.getenv("DIFY_API_KEY")  # ลองใช้ App Key ก่อน
    base_url = os.getenv("DIFY_API_URL", "http://localhost")
    dataset_id = os.getenv("DIFY_DATASET_ID")
    
    print("="*60)
    print("🧪 Testing Dify Connection")
    print("="*60)
    print(f"🔑 API Key: {api_key[:15]}...{api_key[-5:]}")
    print(f"🌐 Base URL: {base_url}")
    print(f"📦 Dataset ID: {dataset_id}")
    print()
    
    # Test 1: ดึงรายชื่อ datasets
    print("📋 Test 1: List all datasets")
    url1 = f"{base_url}/v1/datasets"
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(
                url1,
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=30
            )
            print(f"   Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                datasets = data.get('data', [])
                print(f"   ✅ Found {len(datasets)} datasets")
                
                for ds in datasets[:3]:  # แสดง 3 ตัวแรก
                    print(f"      - {ds.get('name')} (ID: {ds.get('id')})")
            else:
                print(f"   ❌ Error: {response.text}")
        except Exception as e:
            print(f"   ❌ Exception: {e}")
    
    print()
    
    # Test 2: ดึง documents จาก dataset
    print("📄 Test 2: Get documents from dataset")
    url2 = f"{base_url}/v1/datasets/{dataset_id}/documents"
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(
                url2,
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=30
            )
            print(f"   Status: {response.status_code}")
            print(f"   Response: {response.text[:200]}...")
            
            if response.status_code == 200:
                data = response.json()
                docs = data.get('data', [])
                print(f"   ✅ Found {len(docs)} documents")
                
                if docs:
                    print(f"\n   📝 First document:")
                    doc = docs[0]
                    print(f"      ID: {doc.get('id')}")
                    print(f"      Name: {doc.get('name')}")
                    print(f"      Word count: {doc.get('word_count', 0)}")
                else:
                    print(f"   ⚠️  Dataset is empty!")
                    print(f"   💡 Go to Dify and upload some documents first")
            else:
                print(f"   ❌ Error: {response.text}")
                
        except Exception as e:
            print(f"   ❌ Exception: {e}")
    
    print()
    print("="*60)

if __name__ == "__main__":
    asyncio.run(test())