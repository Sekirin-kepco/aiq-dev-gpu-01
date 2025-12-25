"""
OCR PoC script: extract images from PDFs with PyMuPDF, apply simple preprocessing,
and run OCR with pytesseract if available.

Usage: python3 tools/ocr_poc.py --dir ../sample_docs --out results.json
"""
import argparse
import json
import os
import sys
import traceback

try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

try:
    from PIL import Image, ImageFilter, ImageOps
except Exception:
    Image = None

try:
    import pytesseract
except Exception:
    pytesseract = None


def preprocess_image_pil(img: Image.Image) -> Image.Image:
    # Convert to grayscale, increase contrast, and apply slight sharpening
    img = img.convert('L')
    img = ImageOps.autocontrast(img)
    img = img.filter(ImageFilter.SHARPEN)
    return img


def ocr_image(img: Image.Image) -> str:
    if pytesseract is None:
        raise RuntimeError('pytesseract not installed in this environment')
    return pytesseract.image_to_string(img, lang='jpn')


def process_pdf(pdf_path: str) -> dict:
    out = {'file': pdf_path, 'pages': []}
    if fitz is None:
        out['error'] = 'PyMuPDF (fitz) not available'
        return out
    try:
        doc = fitz.open(pdf_path)
        for i, page in enumerate(doc):
            pix = page.get_pixmap(dpi=300)
            mode = 'RGBA' if pix.alpha else 'RGB'
            img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
            if Image is not None:
                imgp = preprocess_image_pil(img)
            else:
                imgp = img
            text = ''
            try:
                text = ocr_image(imgp)
            except Exception as e:
                text = f'[ocr_error] {e}'
            out['pages'].append({'page': i+1, 'text_snippet': text[:500]})
        return out
    except Exception as e:
        return {'file': pdf_path, 'error': str(e)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', required=True, help='Directory containing PDF files')
    parser.add_argument('--out', required=False, default='ocr_results.json')
    args = parser.parse_args()

    results = []
    for root, dirs, files in os.walk(args.dir):
        for fn in files:
            if fn.lower().endswith('.pdf'):
                path = os.path.join(root, fn)
                print('Processing', path)
                try:
                    res = process_pdf(path)
                except Exception:
                    res = {'file': path, 'error': traceback.format_exc()}
                results.append(res)

    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print('Wrote', args.out)


if __name__ == '__main__':
    main()
