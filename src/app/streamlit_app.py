"""
Streamlit 前端：工地安全多模态助手（类似 DeepSeek 的聊天界面）。

功能概览：
- 左侧：会话配置（选择模型 Provider、是否启用视觉、多轮对话开关等）；
- 中间：对话区，支持文本消息与图片（上传或从数据集中选取预览）；
- 下方：输入框 + 发送按钮；
- 右侧（可选）：数据集与标注状态简要预览（占位，后续与数据管线打通）。

说明：
- 这里只实现前端交互与基础回调逻辑，真正的推理/标注由 src.data 下的模块提供；
- 为了保证本仓库可运行，默认后端调用使用“假响应”，方便你后续替换为真实 VLM API。
"""

from __future__ import annotations

import base64
import os
from io import BytesIO
from typing import List, Dict, Any, Optional

import streamlit as st
from PIL import Image
from dashscope import MultiModalConversation
import dashscope

from src.data.download_helmet_dataset import prepare_shwd_dataset
from src.data.auto_annotate_vlm import auto_annotate_directory
from src.config.paths import PATHS


def encode_image_to_base64(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def init_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages: List[Dict[str, Any]] = []
    if "current_image_b64" not in st.session_state:
        st.session_state.current_image_b64: Optional[str] = None


def call_qwen_vl_chat(text: str, image_b64: Optional[str] = None, model: str = "qwen-vl-max") -> str:
    """
    真正调用 Qwen-VL API 进行对话。
    
    Args:
        text: 用户的问题
        image_b64: base64 编码的图片（可选）
        model: 使用的模型名称
    
    Returns:
        模型生成的回复
    """
    # 从 session_state 获取 API key
    api_key = st.session_state.get("dashscope_api_key") or os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        return "错误：请先在左侧边栏设置 DASHSCOPE_API_KEY"
    
    dashscope.api_key = api_key
    
    # 构建系统提示词
    system_prompt = """你是一个专门负责工地安全隐患识别的视觉助手，擅长判断人员是否佩戴安全帽、是否穿反光衣，
以及现场是否存在临边防护不足、杂物堆放、高处坠落风险等问题。
请根据用户的问题，结合图片（如有）给出专业的安全分析和建议。
请用中文回答，语气专业、严谨。"""
    
    # 构建消息
    if image_b64:
        # 解码 base64 图片
        img_bytes = base64.b64decode(image_b64)
        
        # 对于 DashScope，需要使用文件 URL 或本地文件路径
        # 这里我们使用通用图像格式
        messages = [
            {
                "role": "system",
                "content": [{"text": system_prompt}],
            },
            {
                "role": "user",
                "content": [
                    {"image": "https://dummy-image-placeholder.com/placeholder.jpg"},
                    {"text": text}
                ],
            },
        ]
        
        # 由于 DashScope 需要文件 URL，我们先将图片保存为临时文件
        import tempfile
        import uuid
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp.write(img_bytes)
            tmp_path = tmp.name
        
        try:
            messages = [
                {
                    "role": "system",
                    "content": [{"text": system_prompt}],
                },
                {
                    "role": "user",
                    "content": [
                        {"image": f"file://{tmp_path}"},
                        {"text": text}
                    ],
                },
            ]
            
            resp = MultiModalConversation.call(model=model, messages=messages)
            
            # 解析响应
            content_items = (
                getattr(resp, "output", None)
                and getattr(resp.output, "choices", None)
                and resp.output.choices[0].message.content
            )
            
            if not content_items:
                return f"API 返回空响应：{resp}"
            
            texts = [c.get("text", "") for c in content_items if isinstance(c, dict)]
            result = "\n".join(t.strip() for t in texts if t.strip())
            
            return result if result else "模型未返回有效响应"
            
        finally:
            # 清理临时文件
            try:
                os.unlink(tmp_path)
            except:
                pass
    else:
        # 无图片的文本对话
        messages = [
            {
                "role": "system",
                "content": [{"text": system_prompt}],
            },
            {
                "role": "user",
                "content": [{"text": text}],
            },
        ]
        
        resp = MultiModalConversation.call(model=model, messages=messages)
        
        content_items = (
            getattr(resp, "output", None)
            and getattr(resp.output, "choices", None)
            and resp.output.choices[0].message.content
        )
        
        if not content_items:
            return f"API 返回空响应：{resp}"
        
        texts = [c.get("text", "") for c in content_items if isinstance(c, dict)]
        result = "\n".join(t.strip() for t in texts if t.strip())
        
        return result if result else "模型未返回有效响应"


def fake_vlm_response(text: str, has_image: bool) -> str:
    """
    占位推理函数：为了让前端可以跑通，在未接好真实大模型时使用。
    后续可以在此处调用自定义的 VLM 推理服务。
    """
    if has_image:
        prefix = "【视觉+文本分析占位响应】"
        detail = (
            "我会根据上传的工地图片识别安全帽、反光衣、临边防护等情况，并指出潜在隐患。"
        )
    else:
        prefix = "【文本分析占位响应】"
        detail = "我会基于工地安全规范，从人员防护、临边防护、材料堆放等角度给出建议。"
    return f"{prefix}\n你问的问题是：{text}\n\n{detail}\n\n（当前为假响应，请接入真实视觉大模型）"


def render_sidebar() -> Dict[str, Any]:
    st.sidebar.title("⚙️ 会话配置")
    st.sidebar.caption("面向工地安全场景的多模态助手")

    # API Key 配置
    st.sidebar.subheader("🔑 API 配置")
    
    # 初始化 session_state 中的 api_key
    if "dashscope_api_key" not in st.session_state:
        st.session_state.dashscope_api_key = "sk-3245134be4cd43d5990710278f853606"
    
    api_key_input = st.sidebar.text_input(
        "DASHSCOPE API Key", 
        type="password",
        value=st.session_state.dashscope_api_key,
        help="输入阿里云 DashScope 的 API Key"
    )
    if api_key_input:
        st.session_state.dashscope_api_key = api_key_input
        os.environ["DASHSCOPE_API_KEY"] = api_key_input
        dashscope.api_key = api_key_input  # 立即生效
    
    # 模型选择
    model = st.sidebar.selectbox(
        "选择 Qwen-VL 模型",
        ["qwen-vl-max", "qwen-vl-plus"],
        index=0,
        help="qwen-vl-max 效果更好，qwen-vl-plus 更快更便宜"
    )
    
    enable_vision = st.sidebar.checkbox("启用视觉理解（图像 + 文本）", value=True)
    multi_turn = st.sidebar.checkbox("多轮对话（记忆上下文）", value=True)

    st.sidebar.markdown("---")
    st.sidebar.subheader("数据集 & 标注")
    st.sidebar.caption("一键下载安全帽数据集，并使用 Qwen-VL 自动标注。")

    dataset_action = st.sidebar.selectbox(
        "数据操作",
        ["无操作", "下载示例工地安全数据集", "运行自动标注（VLM）"],
        index=0,
    )

    dataset_status: Optional[str] = None

    if dataset_action == "下载示例工地安全数据集":
        if st.sidebar.button("开始下载 SHWD 数据集", use_container_width=True):
            st.sidebar.info("正在下载并解压 SHWD 安全帽数据集...")
            try:
                jpeg_dir = prepare_shwd_dataset()
                dataset_status = f"✅ 数据集已就绪：{jpeg_dir}"
            except Exception as e:
                dataset_status = f"❌ 下载失败：{e}"

    elif dataset_action == "运行自动标注（VLM）":
        st.sidebar.caption("使用千问 Qwen-VL 对安全帽图片进行自动标注（JSONL）。")
        model = st.sidebar.selectbox(
            "选择 Qwen-VL 模型", ["qwen-vl-max", "qwen-vl-plus"], index=0
        )
        max_images = st.sidebar.number_input(
            "最多标注图片数量", min_value=1, max_value=2000, value=200, step=50
        )
        if st.sidebar.button("开始自动标注", use_container_width=True):
            # 使用sidebar外的info提示
            st.sidebar.info("正在调用 Qwen-VL 进行自动标注，请稍候...")
            try:
                from src.config.paths import PATHS  # 局部导入避免循环

                default_image_dir = os.path.join(
                    PATHS.raw_data_dir, "helmet", "kaggle", "images"
                )
                if not os.path.exists(default_image_dir):
                    raise FileNotFoundError(
                          f"未找到图片目录 {default_image_dir}，请先执行'下载示例工地安全数据集'。"
                    )

                output_path = os.path.join(
                    PATHS.processed_data_dir,
                    "helmet_qwen_annotations_streamlit.jsonl",
                )
                auto_annotate_directory(
                    image_dir=default_image_dir,
                    output_path=output_path,
                    model=model,
                    max_images=int(max_images),
                )
                dataset_status = f"✅ 自动标注完成，结果已保存到：{output_path}"
            except Exception as e:
                dataset_status = f"❌ 自动标注失败：{e}"

    if dataset_status:
        if dataset_status.startswith("✅"):
            st.sidebar.success(dataset_status)
        else:
            st.sidebar.error(dataset_status)

    return {
        "model": model,
        "enable_vision": enable_vision,
        "multi_turn": multi_turn,
        "dataset_action": dataset_action,
    }


def render_chat_header() -> None:
    st.markdown(
        """
        <div style="display:flex;align-items:center;gap:0.75rem;">
          <div style="width:36px;height:36px;border-radius:999px;background:linear-gradient(135deg,#0f172a,#38bdf8);display:flex;align-items:center;justify-content:center;color:white;font-weight:600;font-size:20px;">
            S
          </div>
          <div>
            <div style="font-size:1.1rem;font-weight:600;">工地安全多模态助手</div>
            <div style="font-size:0.85rem;color:#64748b;">上传工地现场图片，结合规范快速发现安全隐患</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")


def render_messages() -> None:
    for msg in st.session_state.messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        image_b64 = msg.get("image_b64")

        if role == "user":
            with st.chat_message("user", avatar="👷"):
                st.markdown(content)
                if image_b64:
                    img_bytes = base64.b64decode(image_b64)
                    img = Image.open(BytesIO(img_bytes))
                    st.image(img, caption="用户上传的工地图片")
        else:
            with st.chat_message("assistant", avatar="🛡️"):
                st.markdown(content)


def main() -> None:
    st.set_page_config(
        page_title="工地安全多模态助手",
        page_icon="🛡️",
        layout="wide",
    )
    init_session_state()

    cfg = render_sidebar()

    left, right = st.columns([3, 2])

    with left:
        render_chat_header()
        render_messages()

        # 底部输入区
        st.markdown("### 提问")

        with st.container():
            img_file = st.file_uploader(
                "上传工地图片（可选）", type=["jpg", "jpeg", "png", "bmp", "webp"]
            )
            if img_file is not None and cfg["enable_vision"]:
                img = Image.open(img_file).convert("RGB")
                st.image(img, caption="已选图片预览")
                st.session_state.current_image_b64 = encode_image_to_base64(img)

            prompt = st.text_area(
                "输入你的问题，例如：这张图中有哪些安全隐患？",
                height=80,
                key="chat_input",
            )

            col1, col2 = st.columns([1, 4])
            with col1:
                send_clicked = st.button("发送", type="primary", use_container_width=True)
            with col2:
                clear_clicked = st.button(
                    "清空对话", type="secondary", use_container_width=True
                )

        if clear_clicked:
            st.session_state.messages = []
            st.session_state.current_image_b64 = None
            st.rerun()

        if send_clicked and prompt.strip():
            image_b64 = (
                st.session_state.current_image_b64 if cfg["enable_vision"] else None
            )

            # 记录用户消息
            st.session_state.messages.append(
                {"role": "user", "content": prompt.strip(), "image_b64": image_b64}
            )

            # 调用真实 Qwen-VL API 生成响应
            has_image = image_b64 is not None
            reply = call_qwen_vl_chat(prompt.strip(), image_b64=image_b64, model=cfg["model"])
            st.session_state.messages.append(
                {"role": "assistant", "content": reply, "image_b64": None}
            )

            st.rerun()

    with right:
        # 数据统计功能
        def get_dataset_stats():
            from src.config.paths import PATHS
            import json
            import os
            
            stats = {
                "raw_images": 0,
                "vlm_annotations": 0,
                "vqa_samples": 0
            }
            
            # 原始图片数量
            img_dir = os.path.join(PATHS.raw_data_dir, "helmet", "kaggle", "images")
            if os.path.exists(img_dir):
                stats["raw_images"] = len([f for f in os.listdir(img_dir) if f.endswith('.png')])
            
            # VLM标注数量 (合并所有标注文件)
            processed_dir = os.path.join(PATHS.processed_data_dir)
            if os.path.exists(processed_dir):
                for f in os.listdir(processed_dir):
                    if f.endswith('.jsonl'):
                        filepath = os.path.join(processed_dir, f)
                        with open(filepath, 'r', encoding='utf-8') as file:
                            stats["vlm_annotations"] += sum(1 for _ in file)
            
            # VQA样本数量
            vqa_file = os.path.join(PATHS.vqa_data_dir, "train.jsonl")
            if os.path.exists(vqa_file):
                with open(vqa_file, 'r', encoding='utf-8') as file:
                    stats["vqa_samples"] = sum(1 for _ in file)
            
            return stats
        
        st.markdown("### 📊 数据统计")
        
        stats = get_dataset_stats()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("📷 原始图片", stats["raw_images"])
            st.metric("📝 VQA样本", stats["vqa_samples"])
        with col2:
            st.metric("🤖 VLM标注", stats["vlm_annotations"])
        
        # 进度条显示
        vqa_progress = min(stats["vqa_samples"] / 500, 1.0)  # 目标500条
        st.progress(vqa_progress, text=f"VQA数据构建进度: {stats['vqa_samples']}/500")
        
        # 状态指示
        if stats["vqa_samples"] >= 300:
            st.success("✅ VQA数据集已满足最低要求(300条)")
        else:
            st.warning(f"⚠️ VQA数据集还需补充({stats['vqa_samples']}/300)")


if __name__ == "__main__":
    main()

