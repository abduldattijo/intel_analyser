#!/usr/bin/env python3
"""
Test file for Intelligence Processor with Analytics
Run this to verify your setup is working correctly
"""

import sys
import os
from datetime import datetime

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))


def test_imports():
    """Test that all required modules can be imported"""
    print("🧪 Testing imports...")

    try:
        import spacy
        print("✅ spaCy imported successfully")
    except ImportError as e:
        print(f"❌ spaCy import failed: {e}")
        return False

    try:
        import streamlit
        print("✅ Streamlit imported successfully")
    except ImportError as e:
        print(f"❌ Streamlit import failed: {e}")
        return False

    try:
        import pandas
        print("✅ Pandas imported successfully")
    except ImportError as e:
        print(f"❌ Pandas import failed: {e}")
        return False

    try:
        import networkx
        print("✅ NetworkX imported successfully")
    except ImportError as e:
        print(f"❌ NetworkX import failed: {e}")
        return False

    try:
        import plotly
        print("✅ Plotly imported successfully")
    except ImportError as e:
        print(f"❌ Plotly import failed: {e}")
        return False

    return True


def test_spacy_model():
    """Test that spaCy language model is available"""
    print("\n🧪 Testing spaCy language model...")

    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        print("✅ spaCy English model loaded successfully")

        # Test basic processing
        doc = nlp("Apple Inc. is based in Cupertino, California.")
        entities = [ent.text for ent in doc.ents]
        print(f"✅ Entity extraction test: found {len(entities)} entities: {entities}")
        return True

    except OSError:
        print("❌ spaCy English model not found")
        print("💡 Run: python -m spacy download en_core_web_sm")
        return False
    except Exception as e:
        print(f"❌ spaCy model test failed: {e}")
        return False


def test_ocr_integration():
    """Test OCR functionality"""
    print("\n🔍 Testing OCR integration...")

    try:
        import pytesseract
        from PIL import Image
        import pdf2image
        print("✅ OCR libraries imported successfully")
    except ImportError as e:
        print(f"⚠️ OCR libraries not available: {e}")
        print("💡 Install with: brew install tesseract && pip install pytesseract Pillow pdf2image")
        return False

    try:
        # Test Tesseract
        version = pytesseract.get_tesseract_version()
        print(f"✅ Tesseract version: {version}")
    except Exception as e:
        print(f"❌ Tesseract not properly installed: {e}")
        print("💡 Install with: brew install tesseract")
        return False

    return True


def test_analytics_engine():
    """Test the AnalyticsEngine class"""
    print("\n🧠 Testing AnalyticsEngine...")

    try:
        from analytics_engine import AnalyticsEngine
        print("✅ AnalyticsEngine imported successfully")

        # Create analytics engine
        analytics = AnalyticsEngine()
        print("✅ AnalyticsEngine created successfully")

        # Test sample data for analytics
        sample_reports = [
            {
                'doc_id': 'TEST-001',
                'doc_type': 'security_intelligence',
                'text': 'On June 5, 2025, JOHN SMITH was observed in MOSCOW meeting with VLADIMIR PETROV. This is a critical threat requiring immediate action.',
                'processed_date': datetime.now()
            }
        ]

        sample_entities = [
            {'text': 'MOSCOW', 'label': 'GPE', 'doc_id': 'TEST-001', 'doc_type': 'security_intelligence'},
            {'text': 'JOHN SMITH', 'label': 'PERSON', 'doc_id': 'TEST-001', 'doc_type': 'security_intelligence'}
        ]

        # Test timeline analysis
        print("🔄 Testing timeline analysis...")
        timeline_result = analytics.analyze_timeline(sample_reports)
        if 'error' not in timeline_result:
            print("✅ Timeline analysis working")
        else:
            print(f"⚠️ Timeline analysis: {timeline_result['error']}")

        # Test geographic analysis
        print("🔄 Testing geographic analysis...")
        geo_result = analytics.analyze_geography(sample_entities)
        if 'error' not in geo_result:
            print("✅ Geographic analysis working")
        else:
            print(f"⚠️ Geographic analysis: {geo_result['error']}")

        # Test sentiment analysis
        print("🔄 Testing sentiment analysis...")
        sentiment_result = analytics.analyze_sentiment(sample_reports)
        if 'error' not in sentiment_result:
            print("✅ Sentiment analysis working")
        else:
            print(f"⚠️ Sentiment analysis: {sentiment_result['error']}")

        return True

    except Exception as e:
        print(f"❌ AnalyticsEngine test failed: {e}")
        return False


def test_analytics_libraries():
    """Test analytics library imports"""
    print("\n📊 Testing analytics libraries...")

    libraries = {
        'folium': 'Geographic mapping',
        'textblob': 'Sentiment analysis',
        'vaderSentiment': 'Advanced sentiment analysis',
        'geopy': 'Geocoding services',
        'dateparser': 'Date parsing',
        'wordcloud': 'Word cloud generation'
    }

    available_count = 0
    for lib, description in libraries.items():
        try:
            __import__(lib)
            print(f"✅ {lib}: {description}")
            available_count += 1
        except ImportError:
            print(f"❌ {lib}: {description} - not available")

    print(f"📊 Analytics libraries: {available_count}/{len(libraries)} available")
    return available_count == len(libraries)


def test_document_processor():
    """Test the DocumentProcessor class"""
    print("\n🧪 Testing DocumentProcessor...")

    try:
        from document_processor import DocumentProcessor
        print("✅ DocumentProcessor imported successfully")

        # Create processor instance
        doc_processor = DocumentProcessor()
        print("✅ DocumentProcessor created successfully")

        # Check supported formats
        supported_formats = [fmt for fmt, supported in doc_processor.supported_formats.items() if supported]
        print(f"✅ Supported formats: {supported_formats}")

        # Check OCR support
        ocr_formats = [fmt for fmt in supported_formats if fmt in ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']]
        if ocr_formats:
            print(f"🔍 OCR formats available: {ocr_formats}")
        else:
            print("⚠️ No OCR image formats detected")

        # Check for missing dependencies
        missing_deps = doc_processor.get_missing_dependencies()
        if missing_deps:
            print("⚠️ Missing dependencies:")
            for format_name, install_cmd in missing_deps.items():
                print(f"   {format_name}: {install_cmd}")
        else:
            print("✅ All document processing libraries available")

        # Test text extraction with sample content
        import io

        # Test TXT processing
        sample_txt = "INTELLIGENCE REPORT\nSubject: Test Analysis\nJOHN SMITH from ACME CORP was seen meeting with JANE DOE."
        txt_file = io.BytesIO(sample_txt.encode('utf-8'))
        txt_file.name = "test.txt"

        extracted_text, metadata = doc_processor.extract_text(txt_file)
        print(f"✅ TXT extraction test: {len(extracted_text)} characters extracted")
        print(f"   Method: {metadata['extraction_method']}")

        return True

    except Exception as e:
        print(f"❌ DocumentProcessor test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("🔬 Running Intelligence Analysis POC Tests with Advanced Analytics")
    print("=" * 70)

    all_passed = True

    all_passed &= test_imports()
    all_passed &= test_spacy_model()
    all_passed &= test_processor()
    all_passed &= test_document_processor()
    all_passed &= test_ocr_integration()
    all_passed &= test_analytics_libraries()
    all_passed &= test_analytics_engine()

    print("\n" + "=" * 70)
    if all_passed:
        print("🎉 All tests passed! Your advanced analytics system is ready.")
        print("🚀 You can now run: python run_dashboard.py")
        print("\n💡 New Analytics Features Available:")
        print("   📅 Timeline Analysis - Track intelligence evolution over time")
        print("   🗺️ Geographic Mapping - Visualize locations mentioned in documents")
        print("   📊 Sentiment Analysis - Assess threat levels and urgency")
        print("\n🧪 To test analytics:")
        print("   python create_ocr_samples.py  # Creates sample images")
        print("   Then process documents and explore the new analytics tabs!")
    else:
        print("❌ Some tests failed. Please check the error messages above.")
        print("💡 Make sure you've installed:")
        print("   • All Python requirements: pip install -r requirements.txt")
        print("   • spaCy model: python -m spacy download en_core_web_sm")
        print("   • Tesseract OCR: brew install tesseract")
        print("   • TextBlob corpora: python -c \"import nltk; nltk.download('punkt'); nltk.download('brown')\"")

    return all_passed


if __name__ == "__main__":
    main()