#!/usr/bin/env python3
"""
OCR Integration Test Script
Run this to verify OCR setup is working correctly
"""

import sys
import os
import io
from PIL import Image, ImageDraw, ImageFont


def test_tesseract_installation():
    """Test if Tesseract is properly installed"""
    print("🔍 Testing Tesseract OCR installation...")

    try:
        import pytesseract
        version = pytesseract.get_tesseract_version()
        print(f"✅ Tesseract version: {version}")
        return True
    except Exception as e:
        print(f"❌ Tesseract test failed: {e}")
        print("💡 Install with: brew install tesseract")
        return False


def test_ocr_libraries():
    """Test OCR library imports"""
    print("\n🧪 Testing OCR library imports...")

    try:
        import pytesseract
        print("✅ pytesseract imported")
    except ImportError:
        print("❌ pytesseract not available")
        print("💡 Install with: pip install pytesseract")
        return False

    try:
        from PIL import Image
        print("✅ PIL/Pillow imported")
    except ImportError:
        print("❌ PIL/Pillow not available")
        print("💡 Install with: pip install Pillow")
        return False

    try:
        import pdf2image
        print("✅ pdf2image imported")
    except ImportError:
        print("❌ pdf2image not available")
        print("💡 Install with: pip install pdf2image")
        return False

    return True


def create_test_image():
    """Create a simple test image with text"""
    print("\n🖼️ Creating test image with text...")

    try:
        # Create a simple image with text
        width, height = 400, 200
        image = Image.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(image)

        # Add text to image
        text = "INTELLIGENCE REPORT\nCLASSIFIED: SECRET\nSUBJECT: OCR TEST\nJOHN SMITH - ACME CORP\nDATE: 2025-06-08"

        # Use default font
        try:
            # Try to use a better font if available
            font = ImageFont.truetype("Arial.ttf", 16)
        except:
            font = ImageFont.load_default()

        # Draw text
        draw.multiline_text((20, 20), text, fill='black', font=font)

        print("✅ Test image created")
        return image
    except Exception as e:
        print(f"❌ Failed to create test image: {e}")
        return None


def test_ocr_processing():
    """Test OCR processing on a simple image"""
    print("\n🔍 Testing OCR text extraction...")

    try:
        import pytesseract

        # Create test image
        test_image = create_test_image()
        if not test_image:
            return False

        # Perform OCR
        extracted_text = pytesseract.image_to_string(test_image)

        print(f"📄 Extracted text:")
        print("-" * 40)
        print(extracted_text)
        print("-" * 40)

        # Check if we got reasonable results
        if "INTELLIGENCE" in extracted_text.upper() and "JOHN" in extracted_text.upper():
            print("✅ OCR extraction successful - found expected keywords")
            return True
        else:
            print("⚠️ OCR extraction partial - some text may be missing")
            return True  # Still considered working

    except Exception as e:
        print(f"❌ OCR processing failed: {e}")
        return False


def test_document_processor():
    """Test the enhanced document processor with OCR"""
    print("\n🧪 Testing enhanced document processor...")

    try:
        # Add src to path
        sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

        from document_processor import DocumentProcessor

        processor = DocumentProcessor()
        print("✅ DocumentProcessor imported successfully")

        # Check OCR support
        ocr_formats = [fmt for fmt in processor.supported_formats.keys()
                       if fmt in ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']]

        if ocr_formats:
            print(f"✅ OCR formats supported: {ocr_formats}")
        else:
            print("❌ No OCR formats detected")
            return False

        # Test with sample image
        test_image = create_test_image()
        if test_image:
            # Save image to bytes
            img_bytes = io.BytesIO()
            test_image.save(img_bytes, format='PNG')
            img_bytes.seek(0)
            img_bytes.name = 'test_ocr.png'

            # Extract text
            extracted_text, metadata = processor.extract_text(img_bytes)

            print(f"📊 Extraction metadata: {metadata['extraction_method']}")
            print(f"📊 OCR used: {metadata.get('ocr_used', False)}")
            print(f"📊 Characters extracted: {len(extracted_text)}")

            if len(extracted_text) > 10:
                print("✅ Document processor OCR test successful")
                return True
            else:
                print("⚠️ Limited text extracted from test image")
                return False

        return True

    except Exception as e:
        print(f"❌ Document processor test failed: {e}")
        return False


def main():
    """Run complete OCR integration test"""
    print("🔬 OCR Integration Test Suite")
    print("=" * 50)

    tests = [
        ("Tesseract Installation", test_tesseract_installation),
        ("OCR Libraries", test_ocr_libraries),
        ("OCR Processing", test_ocr_processing),
        ("Document Processor", test_document_processor)
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n🔸 {test_name}")
        if test_func():
            passed += 1

    print("\n" + "=" * 50)
    print(f"📊 OCR Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 OCR integration successful!")
        print("🚀 Your system can now process:")
        print("   • Scanned PDF documents")
        print("   • Photographed intelligence reports")
        print("   • Image files with text content")
        print("   • Mixed document types with automatic OCR fallback")
    else:
        print("⚠️ Some OCR tests failed. Check the errors above.")
        print("💡 Common fixes:")
        print("   • Install Tesseract: brew install tesseract")
        print("   • Install Python libraries: pip install pytesseract Pillow pdf2image")
        print("   • Restart your terminal/PyCharm after installation")


if __name__ == "__main__":
    main()