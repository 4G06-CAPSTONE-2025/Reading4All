"""
PDF Image & Caption Extractor for Textbooks
Extracts all images from a PDF with associated captions and page numbers
"""

import os
import re
import csv
import json
import requests
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import fitz  # PyMuPDF
from PIL import Image
import io

class PDFImageExtractor:
    """Extract images and captions from PDF textbooks"""
    
    def __init__(self, pdf_path: str, output_dir: str = "extracted_textbook"):
        """
        Initialize the extractor
        
        Args:
            pdf_path: Path to PDF file or URL
            output_dir: Directory to save extracted images and CSV
        """
        self.pdf_path = pdf_path
        self.output_dir = Path(output_dir)
        self.images_dir = self.output_dir / "images"
        self.doc = None
        
        # Create output directories
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.images_dir.mkdir(exist_ok=True, parents=True)
        
        # Statistics
        self.stats = {
            'total_pages': 0,
            'images_extracted': 0,
            'captions_found': 0,
            'errors': 0
        }
    
    def download_pdf(self, url: str) -> str:
        """Download PDF from URL"""
        print(f"\nDownloading PDF from: {url}")
        
        local_path = self.output_dir / "textbook.pdf"
        
        if local_path.exists():
            print(f"PDF already downloaded: {local_path}")
            return str(local_path)
        
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 8192
        downloaded = 0
        
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=block_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        print(f"\rDownload progress: {progress:.1f}%", end='')
        
        print(f"\n Downloaded to: {local_path}")
        return str(local_path)
    
    def open_pdf(self):
        """Open the PDF document"""
        # Download if URL
        if self.pdf_path.startswith('http'):
            self.pdf_path = self.download_pdf(self.pdf_path)
        
        print(f"\nOpening PDF: {self.pdf_path}")
        self.doc = fitz.open(self.pdf_path)
        self.stats['total_pages'] = len(self.doc)
        print(f" PDF opened: {self.stats['total_pages']} pages")
    
    def extract_text_around_position(
        self, 
        page: fitz.Page, 
        img_rect: fitz.Rect,
        search_height: int = 100
    ) -> str:
        """
        Extract text near an image position (likely caption)
        
        Args:
            page: PyMuPDF page object
            img_rect: Rectangle coordinates of image
            search_height: Pixels to search below image
            
        Returns:
            Extracted caption text
        """
        # Get page dimensions
        page_height = page.rect.height
        
        # Define search area below the image
        search_rect = fitz.Rect(
            0,  # Left edge
            img_rect.y1,  # Bottom of image
            page.rect.width,  # Right edge
            min(img_rect.y1 + search_height, page_height)  # Below image
        )
        
        # Also search above for some caption styles
        search_rect_above = fitz.Rect(
            0,
            max(img_rect.y0 - 50, 0),
            page.rect.width,
            img_rect.y0
        )
        
        # Extract text from both regions
        text_below = page.get_text("text", clip=search_rect).strip()
        text_above = page.get_text("text", clip=search_rect_above).strip()
        
        # Combine and clean
        caption = ""
        
        # Check for figure/table captions
        caption_patterns = [
            r'Figure\s+\d+[\.:]\s*(.+?)(?=\n\n|\Z)',
            r'Fig\.\s+\d+[\.:]\s*(.+?)(?=\n\n|\Z)',
            r'Table\s+\d+[\.:]\s*(.+?)(?=\n\n|\Z)',
            r'Diagram\s+\d+[\.:]\s*(.+?)(?=\n\n|\Z)',
        ]
        
        # Try below first
        for pattern in caption_patterns:
            match = re.search(pattern, text_below, re.IGNORECASE | re.DOTALL)
            if match:
                caption = match.group(0)
                break
        
        # If not found below, try above
        if not caption:
            for pattern in caption_patterns:
                match = re.search(pattern, text_above, re.IGNORECASE | re.DOTALL)
                if match:
                    caption = match.group(0)
                    break
        
        # If still not found, try to get first few lines
        if not caption:
            lines = text_below.split('\n')
            # Filter out very short lines
            meaningful_lines = [l for l in lines if len(l.strip()) > 10]
            if meaningful_lines:
                caption = meaningful_lines[0]
                # Add second line if first is short
                if len(caption) < 50 and len(meaningful_lines) > 1:
                    caption += " " + meaningful_lines[1]
        
        # Clean up caption
        caption = ' '.join(caption.split())  # Remove extra whitespace
        caption = caption[:500]  # Limit length
        
        return caption
    
    def save_image(
        self,
        image_data: bytes,
        page_num: int,
        img_index: int,
        image_ext: str = "png"
    ) -> str:
        """
        Save image to file
        
        Args:
            image_data: Image bytes
            page_num: Page number
            img_index: Image index on page
            image_ext: Image extension
            
        Returns:
            Relative path to saved image
        """
        filename = f"page{page_num:04d}_img{img_index:03d}.{image_ext}"
        filepath = self.images_dir / filename
        
        try:
            # Convert to PIL Image for better format handling
            pil_image = Image.open(io.BytesIO(image_data))
            
            # Convert to RGB if necessary
            if pil_image.mode in ('RGBA', 'LA', 'P'):
                # Create white background
                background = Image.new('RGB', pil_image.size, (255, 255, 255))
                if pil_image.mode == 'P':
                    pil_image = pil_image.convert('RGBA')
                background.paste(pil_image, mask=pil_image.split()[-1] if pil_image.mode in ('RGBA', 'LA') else None)
                pil_image = background
            elif pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # Save as PNG for quality
            pil_image.save(filepath, 'PNG', optimize=True)
            
            return f"images/{filename}"
            
        except Exception as e:
            print(f"  Error saving image: {e}")
            return None
    
    def extract_images_from_page(
        self,
        page: fitz.Page,
        page_num: int
    ) -> List[Dict]:
        """
        Extract all images from a single page
        
        Args:
            page: PyMuPDF page object
            page_num: Page number (1-indexed)
            
        Returns:
            List of image metadata dicts
        """
        images_data = []
        image_list = page.get_images(full=True)
        
        for img_index, img in enumerate(image_list):
            try:
                xref = img[0]
                
                # Extract image data
                base_image = self.doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                
                # Get image dimensions
                img_width = base_image.get("width", 0)
                img_height = base_image.get("height", 0)
                
                # Filter out very small images (likely icons, logos)
                if img_width < 50 or img_height < 50:
                    continue
                
                # Get image position on page
                img_rect = page.get_image_rects(xref)
                if img_rect:
                    img_rect = img_rect[0]  # First occurrence
                    
                    # Extract caption
                    caption = self.extract_text_around_position(page, img_rect)
                else:
                    caption = ""
                
                # Save image
                image_path = self.save_image(
                    image_bytes,
                    page_num,
                    img_index,
                    image_ext
                )
                
                if image_path:
                    images_data.append({
                        'page_number': page_num,
                        'image_index': img_index,
                        'image_path': image_path,
                        'caption': caption,
                        'width': img_width,
                        'height': img_height
                    })
                    
                    self.stats['images_extracted'] += 1
                    if caption:
                        self.stats['captions_found'] += 1
                
            except Exception as e:
                print(f"  Error extracting image {img_index} from page {page_num}: {e}")
                self.stats['errors'] += 1
        
        return images_data
    
    def extract_all_images(self) -> List[Dict]:
        """
        Extract all images from the PDF
        
        Returns:
            List of all extracted image metadata
        """
        print(f"\n{'='*60}")
        print("Starting image extraction...")
        print(f"{'='*60}\n")
        
        all_images = []
        
        for page_num in range(len(self.doc)):
            actual_page_num = page_num + 1
            page = self.doc[page_num]
            
            print(f"Processing page {actual_page_num}/{self.stats['total_pages']}...", end='')
            
            page_images = self.extract_images_from_page(page, actual_page_num)
            all_images.extend(page_images)
            
            print(f" {len(page_images)} images")
        
        print(f"\n{'='*60}")
        print("Extraction complete!")
        print(f"{'='*60}\n")
        
        return all_images
    
    def create_csv(self, images_data: List[Dict]):
        """
        Create CSV file similar to annotated_physics_data format
        
        Args:
            images_data: List of image metadata dicts
        """
        csv_path = self.output_dir / "extracted_images.csv"
        
        print(f"Creating CSV: {csv_path}")
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            fieldnames = [
                'Diagram-ID',
                'Page-Number',
                'Image-Index',
                'Image-Path',
                'Caption',
                'Width',
                'Height',
                'Modified-Alt-Text'  # Empty - to be filled manually
            ]
            
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for idx, img_data in enumerate(images_data, start=1):
                writer.writerow({
                    'Diagram-ID': idx,
                    'Page-Number': img_data['page_number'],
                    'Image-Index': img_data['image_index'],
                    'Image-Path': img_data['image_path'],
                    'Caption': img_data['caption'],
                    'Width': img_data['width'],
                    'Height': img_data['height'],
                    'Modified-Alt-Text': ''  # To be filled
                })
        
        print(f" CSV created: {csv_path}")
        return csv_path
    
    def create_json(self, images_data: List[Dict]):
        """
        Create JSON file with all metadata
        
        Args:
            images_data: List of image metadata dicts
        """
        json_path = self.output_dir / "extracted_images.json"
        
        print(f"Creating JSON: {json_path}")
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({
                'extraction_date': datetime.now().isoformat(),
                'source_pdf': self.pdf_path,
                'statistics': self.stats,
                'images': images_data
            }, f, indent=2, ensure_ascii=False)
        
        print(f" JSON created: {json_path}")
        return json_path
    
    def create_summary(self):
        """Create summary report"""
        summary_path = self.output_dir / "extraction_summary.txt"
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("PDF Image Extraction Summary\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Source PDF: {self.pdf_path}\n")
            f.write(f"Extraction Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("Statistics:\n")
            f.write(f"  Total Pages: {self.stats['total_pages']}\n")
            f.write(f"  Images Extracted: {self.stats['images_extracted']}\n")
            f.write(f"  Captions Found: {self.stats['captions_found']}\n")
            f.write(f"  Errors: {self.stats['errors']}\n\n")
            f.write(f"Output Directory: {self.output_dir}\n")
            f.write(f"Images Directory: {self.images_dir}\n")
        
        print(f" Summary created: {summary_path}")
        return summary_path
    
    def print_statistics(self):
        """Print extraction statistics"""
        print(f"\n{'='*60}")
        print("Extraction Statistics")
        print(f"{'='*60}")
        print(f"Total Pages:       {self.stats['total_pages']}")
        print(f"Images Extracted:  {self.stats['images_extracted']}")
        print(f"Captions Found:    {self.stats['captions_found']}")
        print(f"Errors:            {self.stats['errors']}")
        print(f"{'='*60}\n")
    
    def extract(self):
        """Run the complete extraction pipeline"""
        try:
            # Open PDF
            self.open_pdf()
            
            # Extract images
            images_data = self.extract_all_images()
            
            # Create outputs
            csv_path = self.create_csv(images_data)
            json_path = self.create_json(images_data)
            summary_path = self.create_summary()
            
            # Print statistics
            self.print_statistics()
            
            # Close PDF
            if self.doc:
                self.doc.close()
            
            print(" Extraction pipeline complete!\n")
            print(f"Output files:")
            print(f"  CSV:     {csv_path}")
            print(f"  JSON:    {json_path}")
            print(f"  Summary: {summary_path}")
            print(f"  Images:  {self.images_dir}")
            
            return {
                'csv_path': csv_path,
                'json_path': json_path,
                'summary_path': summary_path,
                'images_dir': self.images_dir,
                'images_data': images_data
            }
            
        except Exception as e:
            print(f"\n✗ Error during extraction: {e}")
            import traceback
            traceback.print_exc()
            raise

def main():
    """Main function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Extract images and captions from PDF textbooks"
    )
    
    parser.add_argument(
        'pdf_path',
        type=str,
        help='Path to PDF file or URL'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='extracted_textbook',
        help='Output directory (default: extracted_textbook)'
    )
    
    args = parser.parse_args()
    
    # Run extraction
    extractor = PDFImageExtractor(args.pdf_path, args.output)
    extractor.extract()

if __name__ == "__main__":
    # For direct execution
    PDF_URL = "https://assets.openstax.org/oscms-prodcms/media/documents/Physics-WEB_Sab7RrQ.pdf"
    OUTPUT_DIR = "physics_textbook_extracted"
    
    print("="*60)
    print("PDF Image & Caption Extractor")
    print("="*60)
    print(f"\nPDF: {PDF_URL}")
    print(f"Output: {OUTPUT_DIR}\n")
    
    extractor = PDFImageExtractor(PDF_URL, OUTPUT_DIR)
    results = extractor.extract()