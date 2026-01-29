"""
Alt-Text Generation Script for Physics Diagrams using Fine-Tuned Pix2Struct Model

This script provides functionality to generate descriptive alt-text for images, 
specifically physics diagrams, using a fine-tuned Pix2Struct model. It supports
single-image inference, batch processing, and directory-wide processing. 
It also addresses checkpoint loading issues by providing fallback options 
for processor initialization.

Key Features:
- Load a fine-tuned Pix2Struct model and processor.
- Generate alt-text for a single image.
- Generate alt-text for a list of images (batch mode).
- Generate alt-text for all images in a directory, with optional JSON output.
- Configurable generation parameters such as max_length, num_beams, temperature, and sampling.

Classes:
- AltTextGenerator: Encapsulates functionality for alt-text generation.

Command-Line Usage:
- Single image:
    python generate_alt_text.py --model_path <path_to_model> --image <image_path> --output <output_file>
- Directory of images:
    python generate_alt_text.py --model_path <path_to_model> --image_dir <directory_path> --output <output_file>
- Optional arguments for generation:
    --base_model, --max_length, --num_beams, --temperature, --device

Dependencies:
- torch
- transformers
- PIL (Pillow)
- tqdm
- argparse
- json

Hardware:
- Supports GPU acceleration with CUDA if available.

Outputs:
- JSON or JSONL files containing a list of dictionaries with:
    {
        "image_path": "<image filename>",
        "alt_text": "<generated alt-text>"
    }

Example:
    python generate_alt_text.py \
        --model_path ./checkpoints/physics_model \
        --image_dir ./physics_images \
        --output physics_alt_text.json \
        --max_length 768 \
        --num_beams 4 \
        --temperature 1.0 \
        --device cuda
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional
import torch
from PIL import Image
from transformers import Pix2StructProcessor, Pix2StructForConditionalGeneration
from tqdm import tqdm

class AltTextGenerator:
    """Generate alt-text for physics diagrams"""
    
    def __init__(self, model_path: str, base_model: str = None, device: str = "cuda"):
        """
        Initialize the generator
        
        Args:
            model_path: Path to fine-tuned model
            base_model: Base model name for processor (if None, uses model_path)
            device: 'cuda' or 'cpu'
        """
        print(f"\nLoading model from: {model_path}")
        
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        
        # Try to load processor from model_path first, fall back to base model
        try:
            if base_model:
                print(f"Loading processor from base model: {base_model}")
                self.processor = Pix2StructProcessor.from_pretrained(base_model)
            else:
                print("Attempting to load processor from model path...")
                self.processor = Pix2StructProcessor.from_pretrained(model_path)
        except OSError as e:
            print(f"Could not load processor from model path: {e}")
            print("Falling back to default pix2struct-base processor...")
            self.processor = Pix2StructProcessor.from_pretrained("google/pix2struct-base")
        
        # Load model from fine-tuned checkpoint
        print(f"Loading model from: {model_path}")
        self.model = Pix2StructForConditionalGeneration.from_pretrained(model_path)
        
        self.model.to(self.device)
        self.model.eval()
        
        print(f" Model loaded on {self.device}\n")
    
    def generate_alt_text(
        self,
        image_path: str,
        max_length: int = 768,
        num_beams: int = 4,
        temperature: float = 1.0,
        top_p: float = 0.9,
        do_sample: bool = False
    ) -> str:
        """
        Generate alt-text for a single image
        
        Args:
            image_path: Path to image file
            max_length: Maximum length of generated text
            num_beams: Number of beams for beam search
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            do_sample: Whether to use sampling
            
        Returns:
            Generated alt-text string
        """
        # Load and process image
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            raise ValueError(f"Failed to load image {image_path}: {e}")
        
        # Encode image
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        
        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=num_beams,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                early_stopping=True,
                no_repeat_ngram_size=3,
                length_penalty=1.0
            )
        
        # Decode
        alt_text = self.processor.decode(generated_ids[0], skip_special_tokens=True)
        
        return alt_text
    
    def generate_batch(
        self,
        image_paths: List[str],
        output_path: Optional[str] = None,
        **generation_kwargs
    ) -> List[Dict]:
        """
        Generate alt-text for multiple images
        
        Args:
            image_paths: List of image file paths
            output_path: Optional path to save results as JSON
            **generation_kwargs: Additional arguments for generate_alt_text
            
        Returns:
            List of dicts with 'image_path' and 'alt_text'
        """
        results = []
        
        print(f"Generating alt-text for {len(image_paths)} images...\n")
        
        for img_path in tqdm(image_paths):
            try:
                alt_text = self.generate_alt_text(img_path, **generation_kwargs)
                
                results.append({
                    'image_path': str(img_path),
                    'alt_text': alt_text
                })
                
                # Print first few results for monitoring
                if len(results) <= 3:
                    print(f"\nSample output for {Path(img_path).name}:")
                    print(f"  {alt_text[:100]}...")
                
            except Exception as e:
                print(f"\nError processing {img_path}: {e}")
                results.append({
                    'image_path': str(img_path),
                    'alt_text': f"[Error: {str(e)[:100]}]"
                })
        
        # Save results if output path specified
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"\n Results saved to: {output_path}")
        
        return results
    
    def generate_from_directory(
        self,
        image_dir: str,
        output_path: str,
        extensions: List[str] = ['.png', '.jpg', '.jpeg'],
        **generation_kwargs
    ):
        """
        Generate alt-text for all images in a directory
        
        Args:
            image_dir: Directory containing images
            output_path: Path to save results JSON
            extensions: List of valid image extensions
            **generation_kwargs: Additional arguments for generation
        """
        image_dir = Path(image_dir)
        
        # Find all images
        image_paths = []
        for ext in extensions:
            image_paths.extend(image_dir.glob(f"*{ext}"))
            image_paths.extend(image_dir.glob(f"*{ext.upper()}"))
        
        image_paths = sorted(list(set(image_paths)))
        
        if not image_paths:
            print(f"No images found in {image_dir}")
            return []
        
        print(f"Found {len(image_paths)} images in {image_dir}")
        
        return self.generate_batch(image_paths, output_path, **generation_kwargs)

def main():
    parser = argparse.ArgumentParser(description="Generate alt-text for physics diagrams")
    
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to fine-tuned model directory"
    )
    
    parser.add_argument(
        "--base_model",
        type=str,
        default="google/pix2struct-base",
        help="Base model for processor (default: google/pix2struct-base)"
    )
    
    parser.add_argument(
        "--image",
        type=str,
        help="Path to single image file"
    )
    
    parser.add_argument(
        "--image_dir",
        type=str,
        help="Path to directory containing images"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="alt_text_results.json",
        help="Path to save results (default: alt_text_results.json)"
    )
    
    parser.add_argument(
        "--max_length",
        type=int,
        default=768,
        help="Maximum length of generated text (default: 768)"
    )
    
    parser.add_argument(
        "--num_beams",
        type=int,
        default=4,
        help="Number of beams for beam search (default: 4)"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (default: 1.0)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use (default: cuda)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.image and not args.image_dir:
        parser.error("Must provide either --image or --image_dir")
    
    # Initialize generator with base model option
    generator = AltTextGenerator(
        args.model_path, 
        base_model=args.base_model,
        device=args.device
    )
    
    generation_kwargs = {
        'max_length': args.max_length,
        'num_beams': args.num_beams,
        'temperature': args.temperature
    }
    
    # Generate alt-text
    if args.image:
        print(f"Processing single image: {args.image}\n")
        
        alt_text = generator.generate_alt_text(args.image, **generation_kwargs)
        
        print("Generated Alt-Text:")
        print("="*60)
        print(alt_text)
        print("="*60)
        
        # Save result
        result = {
            'image_path': args.image,
            'alt_text': alt_text
        }
        
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"\n Result saved to: {args.output}")
    
    elif args.image_dir:
        generator.generate_from_directory(
            args.image_dir,
            args.output,
            **generation_kwargs
        )
    
    print("\n Done!")

if __name__ == "__main__":
    main()