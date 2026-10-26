import os
from datetime import date
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

# ---------- 读取同目录下的 .env 配置文件 ----------
load_dotenv(Path(__file__).resolve().parent / ".env")

# ---------- 页面配置 ----------
st.set_page_config(page_title="我的智能AI计划助手", page_icon="🧠")
st.title("我的智能AI计划助手")
st.caption(
    "输入想学习的主题，AI 会生成一份带具体日期的 7 天学习计划；"
    "生成后可以继续追问，例如：把第一天的任务改简单点。"
)

# ---------- 从 .env / 环境变量读取配置（默认指向智谱开放平台） ----------
API_KEY = os.getenv("ZHIPUAI_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
BASE_URL = (
    os.getenv("ZHIPUAI_BASE_URL")
    or os.getenv("OPENAI_BASE_URL")
    or "https://open.bigmodel.cn/api/paas/v4/"  # 智谱 AI 的 OpenAI 兼容接口地址
)
MODEL = os.getenv("ZHIPU_MODEL", "glm-4-flash")  # 可通过环境变量 ZHIPU_MODEL 更换模型

# ---------- 时间感知：获取今天的真实日期 ----------
TODAY = date.today()
WEEKDAY_CN = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
TODAY_STR = f"{TODAY.year}年{TODAY.month}月{TODAY.day}日 {WEEKDAY_CN[TODAY.weekday()]}"

# ---------- 系统提示词：告诉大模型今天的日期和输出规则 ----------
SYSTEM_PROMPT = f"""你是一位经验丰富的学习规划专家。今天的真实日期是：{TODAY_STR}。

【首次请求】当用户第一次给出学习主题时，请生成一份详细、可执行的 7 天学习计划，要求：
1. 以今天（{TODAY_STR}）为第 1 天，连续规划 7 天；每一天都必须标注具体日期和星期，
   格式如：第 1 天 · {TODAY.month}月{TODAY.day}日 · {WEEKDAY_CN[TODAY.weekday()]}；
2. 每天列出：学习目标、核心知识点、推荐练习、预计学习时长；
3. 内容由浅入深、循序渐进，最后一天安排综合复习与实战；
4. 使用 Markdown 格式输出，适当使用标题、列表和表格。

【后续修改】当用户要求调整计划时（例如"把第一天的任务改简单点"），
请结合上方聊天记录中的上一版计划进行修改，并输出修改后的完整 7 天计划（保持相同的日期标注格式）。

请始终使用中文回答。
"""

# ---------- 侧边栏：运行信息与控制 ----------
with st.sidebar:
    st.header("⚙️ 运行信息")
    st.write(f"📅 今天：{TODAY_STR}")
    st.write(f"🤖 模型：{MODEL}")
    if st.button("🗑️ 清空对话，重新开始", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ---------- 会话记忆：聊天记录保存在 st.session_state 中，刷新页面前一直保留 ----------
if "messages" not in st.session_state:
    st.session_state.messages = []  # 每条消息形如 {"role": "user"/"assistant", "content": "..."}


def call_model_stream(history):
    """把完整聊天记录发给大模型，流式（逐字）返回回复内容。"""
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": SYSTEM_PROMPT}] + history,
        temperature=0.7,
        max_tokens=4090,
        stream=True,  # 开启流式输出：文字分小块逐段返回
    )
    for chunk in response:
        # 每个 chunk 携带一小段新增文字（delta.content）
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


# ---------- 渲染历史聊天记录 ----------
if not st.session_state.messages:
    st.info("👋 欢迎使用！在下方输入框输入想学习的主题（例如：Python 数据分析），我会为你生成 7 天学习计划。")

for msg in st.session_state.messages:
    avatar = "🙋" if msg["role"] == "user" else "🧠"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

# ---------- 底部聊天输入框：每次发送都会让脚本从头重新运行一遍 ----------
if prompt := st.chat_input("请输入想学习的主题，或继续追问..."):
    print(f"用户输入的内容：{prompt}")  # 在运行 Streamlit 的终端打印用户输入

    # 1. 保存并显示用户这条消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🙋"):
        st.markdown(prompt)

    # 2. 流式生成 AI 回复（打字机效果）
    with st.chat_message("assistant", avatar="🧠"):
        bubble = st.empty()
        if not API_KEY:
            bubble.error(
                "未检测到 API Key。请打开项目目录下的 .env 文件，"
                "在 ZHIPUAI_API_KEY= 后面填入你的智谱 API Key 并保存。"
            )
        else:
            bubble.markdown("正在思考...")  # 先显示思考中，收到第一个字后自动被替换
            full_reply = ""
            try:
                for piece in call_model_stream(st.session_state.messages):
                    full_reply += piece
                    bubble.markdown(full_reply + "▌")  # 逐字追加，带光标的打字机效果
                bubble.markdown(full_reply or "（模型没有返回内容，请重试）")
            except Exception as e:
                bubble.error(f"调用大模型失败：{e}")
                full_reply = ""

            # 3. 把完整回复也存入聊天记录，这样下一轮对话才有"记忆"
            if full_reply:
                st.session_state.messages.append({"role": "assistant", "content": full_reply})

