from dotenv import load_dotenv
import os
from linebot import LineBotApi, WebhookParser

load_dotenv()

LINE_TOKEN = os.getenv("LINE_TOKEN")
DIFY_API_KEY = os.getenv("DIFY_API_KEY")
DIFY_CHAT_URL = os.getenv("DIFY_CHAT_URL")
LINE_SECRET = os.getenv("LINE_SECRET")

line_bot = LineBotApi(LINE_TOKEN)
parser = WebhookParser(LINE_SECRET)