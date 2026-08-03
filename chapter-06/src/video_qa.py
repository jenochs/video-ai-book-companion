"""Video question answering with Qwen3-VL for Chapter 6 of Video Generation with AI (O'Reilly).

This module accompanies Chapter 6, "Vision Language Models for Video
Understanding". It provides the VideoQA class built progressively through the
chapter, plus VideoQAWithGrounding, which parses timestamped responses into
structured TemporalSegment objects using the model's 3D positional encoding
awareness.
"""

import re
from dataclasses import dataclass
from typing import List

import torch


@dataclass
class TemporalSegment:
    """A time segment in a video."""
    start_time: str
    end_time: str
    description: str

    def to_seconds(self, time_str: str) -> float:
        parts = time_str.split(":")
        return int(parts[0]) * 60 + float(parts[1]) if len(parts) == 2 else float(time_str)

    @property
    def duration(self) -> float:
        return self.to_seconds(self.end_time) - self.to_seconds(self.start_time)


class VideoQA:
    """A video question-answering system built on Qwen3-VL."""

    def __init__(self, model_name: str = "Qwen/Qwen3-VL-8B-Instruct"):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map="auto")
        self.processor = AutoProcessor.from_pretrained(model_name)

    def ask(self, video_path: str, question: str, fps: float = 2.0,
            max_frames: int = 32, max_new_tokens: int = 512,
            system_prompt: str = None) -> str:
        """Ask a question about a video."""
        from qwen_vl_utils import process_vision_info

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": [
                {"type": "text", "text": system_prompt}
            ]})
        messages.append({"role": "user", "content": [
            {"type": "video", "video": video_path, "fps": fps, "max_frames": max_frames},
            {"type": "text", "text": question}
        ]})
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages, image_patch_size=16)
        inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs,
                                padding=True, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        return self.processor.decode(
            outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

    def describe(self, video_path: str, **kwargs) -> str:
        """Generate a description of the video."""
        return self.ask(video_path, "Describe what happens in this video.", **kwargs)

    def find_event(self, video_path: str, event: str, **kwargs) -> str:
        """Find when a specific event occurs."""
        return self.ask(video_path, f"When does: {event}? Give timestamps in MM:SS.", **kwargs)


class VideoQAWithGrounding(VideoQA):
    """Extended VideoQA with temporal grounding capabilities."""

    def ground_events(self, video_path: str, query: str, **kwargs) -> List[TemporalSegment]:
        """Find events and return structured temporal segments."""
        system_prompt = """Output events in this format:
[START_TIME-END_TIME] Description
Example: [00:05-00:12] A person enters the door
Use MM:SS format."""

        response = self.ask(video_path, query, system_prompt=system_prompt, **kwargs)

        segments = []
        pattern = r'\[(\d{2}:\d{2})-(\d{2}:\d{2})\]\s*(.+?)(?=\[|$)'
        for start, end, desc in re.findall(pattern, response, re.DOTALL):
            segments.append(TemporalSegment(start.strip(), end.strip(), desc.strip()))
        return segments

    def generate_timeline(self, video_path: str, **kwargs) -> List[TemporalSegment]:
        """Generate a complete timeline of events."""
        return self.ground_events(
            video_path, "List all significant events with timestamps.", **kwargs
        )
