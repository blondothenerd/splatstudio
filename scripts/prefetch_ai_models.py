#!/usr/bin/env python3
import os
from transformers import AutoImageProcessor, AutoModelForDepthEstimation, SegformerForSemanticSegmentation

depth = os.getenv("SPLAT_AI_DEPTH_MODEL", "depth-anything/Depth-Anything-V2-Small-hf")
segment = os.getenv("SPLAT_AI_SEGMENT_MODEL", "nvidia/segformer-b5-finetuned-ade-640-640")
print("Downloading AI models into:", os.environ.get("HF_HOME", "default Hugging Face cache"))
print("Depth:", depth)
AutoImageProcessor.from_pretrained(depth)
AutoModelForDepthEstimation.from_pretrained(depth)
print("Segmentation:", segment)
AutoImageProcessor.from_pretrained(segment)
SegformerForSemanticSegmentation.from_pretrained(segment)
print("AI model cache ready")
