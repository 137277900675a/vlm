#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据整理器 - Qwen2-VL
"""

import torch
from qwen_vl_utils import process_vision_info


class Qwen2VLDataCollator:
    """Qwen2-VL数据整理器"""
    
    def __init__(self, processor):
        self.processor = processor
    
    def __call__(self, features):
        """批量处理数据"""
        # 提取所有messages
        batch_messages = [feature["messages"] for feature in features]
        
        # 应用chat template
        texts = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
            for msg in batch_messages
        ]
        
        # 批量处理视觉信息
        image_inputs, video_inputs = process_vision_info(batch_messages)
        
        # 处理输入
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        
        # 处理标签
        labels = inputs["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        inputs["labels"] = labels
        
        return inputs
