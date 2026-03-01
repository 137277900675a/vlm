#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
推理Demo脚本 - 工地安全VLM
加载LoRA权重进行推理
"""

import os
import json
import argparse
from typing import Optional, Dict, Any

import torch
from PIL import Image
from transformers import Qwen2VLProcessor, Qwen2VLForConditionalGeneration
from peft import PeftModel
import streamlit as st
from src.config.paths import PATHS


def load_inference_model(
    base_model: str = "Qwen/Qwen2-VL-2B-Instruct",
    lora_path: Optional[str] = None,
    device: str = "cuda"
):
    """加载推理模型"""
    # 优先使用本地ModelScope下载的模型
    local_model_path = "models/pretrained/Qwen/Qwen2-VL-2B-Instruct"
    if os.path.exists(local_model_path):
        base_model = local_model_path
        print(f"🔄 使用本地模型: {base_model}")
    else:
        print(f"🔄 加载基础模型: {base_model}")
    
    processor = Qwen2VLProcessor.from_pretrained(base_model)
    
    if torch.cuda.is_available():
        device_map = device
        torch_dtype = torch.bfloat16
    else:
        device_map = "cpu"
        torch_dtype = torch.float32
    
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        base_model,
        torch_dtype=torch_dtype,
        device_map=device_map,
    )
    
    # 加载LoRA权重(如果存在)
    if lora_path and os.path.exists(lora_path):
        print(f"🔄 加载LoRA权重: {lora_path}")
        model = PeftModel.from_pretrained(model, lora_path)
        model = model.merge_and_unload()
    
    model.eval()
    return model, processor


def inference_single_image(
    model,
    processor,
    image_path: str,
    question: str,
    max_new_tokens: int = 256
) -> str:
    """单图推理"""
    from qwen_vl_utils import process_vision_info
    
    # 构建消息
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{os.path.abspath(image_path)}"},
                {"type": "text", "text": question}
            ]
        }
    ]
    
    # 预处理
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    
    # 移动到设备
    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v 
                  for k, v in inputs.items()}
    
    # 推理
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    
    # 解码
    input_length = inputs["input_ids"].shape[1]
    response = processor.batch_decode(
        output[:, input_length:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )[0]
    
    return response


def inference_with_base64(
    model,
    processor,
    image_base64: str,
    question: str,
    max_new_tokens: int = 256
) -> str:
    """使用base64图像进行推理"""
    import base64
    from io import BytesIO
    from qwen_vl_utils import process_vision_info
    
    # 解码base64
    image_data = base64.b64decode(image_base64)
    image = Image.open(BytesIO(image_data)).convert("RGB")
    
    # 构建消息
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question}
            ]
        }
    ]
    
    # 预处理
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    
    # 移动到设备
    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v 
                  for k, v in inputs.items()}
    
    # 推理
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    
    # 解码
    input_length = inputs["input_ids"].shape[1]
    response = processor.batch_decode(
        output[:, input_length:],
        skip_special_tokens=True,
    )[0]
    
    return response


def demo_cli():
    """命令行Demo"""
    parser = argparse.ArgumentParser(description="工地安全VLM推理Demo")
    parser.add_argument("--image", type=str, required=True, help="图像路径")
    parser.add_argument("--question", type=str, required=True, help="问题")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2-VL-2B-Instruct", help="基础模型")
    parser.add_argument("--lora_path", type=str, default=None, help="LoRA权重路径（可选）")
    parser.add_argument("--use_finetuned", action="store_true", help="使用微调模型")
    args = parser.parse_args()
    
    # 如果指定使用微调模型但未提供路径，使用默认路径
    if args.use_finetuned and not args.lora_path:
        args.lora_path = "models/lora_safety"
    # 如果指定使用微调模型但未提供路径，使用默认路径
    if args.use_finetuned and not args.lora_path:
        args.lora_path = "models/lora_safety"
    
    # 加载模型
    model, processor = load_inference_model(args.model, args.lora_path)
    
    # 推理
    print(f"\n📷 图像: {args.image}")
    print(f"❓ 问题: {args.question}")
    print(f"🤖 模型: {'微调后' if args.lora_path else 'Base'}")
    print("\n回答:")
    
    response = inference_single_image(model, processor, args.image, args.question)
    print(response)


def demo_streamlit():
    """Streamlit推理界面"""
    st.set_page_config(
        page_title="工地安全VLM推理Demo",
        page_icon="🏗️",
        layout="wide"
    )
    
    st.title("🏗️ 工地安全VLM推理Demo")
    st.markdown("---")
    
    # 侧边栏 - 模型配置
    with st.sidebar:
        st.header("模型配置")
        
        model_name = st.selectbox(
            "选择基础模型",
            ["Qwen/Qwen2-VL-2B-Instruct"],
            index=0
        )
        
        lora_path = st.text_input(
            "LoRA权重路径(可选)",
            value="models/lora_safety"
        )
        
        if st.button("加载模型"):
            if "model" not in st.session_state:
                with st.spinner("加载模型中..."):
                    model, processor = load_inference_model(model_name, lora_path)
                    st.session_state.model = model
                    st.session_state.processor = processor
                st.success("模型加载完成!")
            else:
                st.info("模型已加载")
    
    # 主界面
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📷 输入图像")
        uploaded_file = st.file_uploader(
            "上传工地图片",
            type=["jpg", "jpeg", "png", "bmp", "webp"]
        )
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="上传的图片", use_container_width=True)
            
            # 保存到临时文件
            temp_path = f"temp_{uploaded_file.name}"
            image.save(temp_path)
            st.session_state.temp_image_path = temp_path
    
    with col2:
        st.subheader("❓ 问答")
        
        # 预设问题
        preset_questions = [
            "请分析这张工地照片中是否存在安全隐患？",
            "图中所有人员是否都佩戴了安全帽？",
            "请列出需要整改的安全问题。"
        ]
        
        question = st.text_area(
            "输入问题",
            value="请分析这张工地照片中是否存在安全隐患？",
            height=100
        )
        
        if st.button("🔍 分析", type="primary"):
            if "model" not in st.session_state:
                st.error("请先在侧边栏加载模型!")
            elif "temp_image_path" not in st.session_state:
                st.error("请先上传图片!")
            else:
                with st.spinner("推理中..."):
                    response = inference_single_image(
                        st.session_state.model,
                        st.session_state.processor,
                        st.session_state.temp_image_path,
                        question
                    )
                    
                    st.success("推理完成!")
                    st.markdown("### 📝 分析结果")
                    st.write(response)
    
    # 预设问题按钮
    st.markdown("### 💡 快速问题")
    cols = st.columns(3)
    for i, q in enumerate(preset_questions):
        if cols[i].button(q[:20] + "..."):
            st.session_state.current_question = q
            st.rerun()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        demo_cli()
    else:
        demo_streamlit()
