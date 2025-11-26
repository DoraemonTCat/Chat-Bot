import requests
from fastapi import FastAPI, Request
from linebot import LineBotApi, WebhookParser
from linebot.models import TextSendMessage


app = FastAPI()

LINE_TOKEN = "OsUuijk64ryTrIijIN6k1sA+NGUkFhGiBRhuYBWRjIweZ5rSkE5AgU4Jk3JfW9X7lWJnhpYF5+/nSvzHkKbXCaS4ZtkbM2u6zU7WlhiV1qrNaLYn+IA68pQghR6r3TAmTHz7q4V9vmIantOekshfjQdB04t89/1O/w1cDnyilFU="
line_bot = LineBotApi(LINE_TOKEN)
parser = WebhookParser("1823b4dcc6c943660a08daa64559a787")

DIFY_CHAT_URL = "http://localhost/v1/chat-messages"
DIFY_API_KEY = "app-aOeOS9EGKdo6plZ1AMG3WUqJ"

@app.post("/webhook")
async def webhook(request: Request):
    body = await request.body()
    events = parser.parse(body.decode('utf-8'), request.headers['X-Line-Signature'])

    for event in events:
        if event.message.type == 'text':
            # ส่งห้อง chatflow ของ Dify
            payload = {
                "query": event.message.text,          # ข้อความผู้ใช้
                "inputs": {},                         # ถ้าไม่มี input variable ใช้ {}
                "response_mode": "blocking",          # หรือ "streaming"
                "user": str(event.source.user_id),    # ใช้ user_id ของผู้ใช้
                # "files": [],                        # ใส่ถ้ามีไฟล์
                # "workflow_id": "wf-xxx",            # ใส่ถ้าอยากใช้ workflow เฉพาะ
            }
            headers = {"Authorization": f"Bearer {DIFY_API_KEY}"}
            res = requests.post(
                DIFY_CHAT_URL,
                headers={"Authorization": f"Bearer {DIFY_API_KEY}", "Content-Type": "application/json"},
                json=payload
            )

            data = res.json()
            reply_text = data.get("answer") or "No response"

            line_bot.reply_message(
                event.reply_token,
                TextSendMessage(text=reply_text)
            )

    return "OK"
