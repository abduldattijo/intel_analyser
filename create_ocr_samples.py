#!/usr/bin/env python3
"""
Create sample images with intelligence text for OCR testing
"""

import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


def create_sample_images():
    """Create sample intelligence images for OCR testing"""

    # Create sample_images directory
    sample_dir = Path("data/sample_images")
    sample_dir.mkdir(parents=True, exist_ok=True)

    # Sample intelligence content for images
    sample_texts = {
        "classified_memo": """CLASSIFIED MEMORANDUM
DATE: June 8, 2025
FROM: Intelligence Division
TO: Operations Team

SUBJECT: Financial Investigation Update

Key entities identified:
- VLADIMIR PETROV (Russian national)
- CRIMSON ENTERPRISES (Shell company)
- SWISS NATIONAL BANK (Financial institution)

Total suspicious transactions: $15.7 million
Geographic focus: CAYMAN ISLANDS, MONACO

Recommend immediate surveillance.

Classification: SECRET""",

        "field_report": """FIELD INTELLIGENCE REPORT
Agent: Sarah Mitchell
Location: Frankfurt, Germany
Date: 2025-06-08

Surveillance target: MARCUS WEBB
Status: Active monitoring

Observed meeting with unknown associate
at EUROPA HOTEL, 14:30 local time.

Exchange of briefcase noted.
License plate: DE-AB-1234

Recommend enhanced surveillance.
Report filed: 15:45""",

        "threat_bulletin": """THREAT ASSESSMENT BULLETIN
Priority: HIGH
Date: 2025-06-08

Cyber threat group APT-COBRA
has increased activities targeting:

- POWER GRID operators
- FINANCIAL INSTITUTIONS  
- TELECOMMUNICATIONS infrastructure

Attack vectors:
1. Spear phishing campaigns
2. Supply chain compromises
3. Zero-day exploits

Immediate action required.
Contact: Cyber Command Center"""
    }

    print("🖼️ Creating OCR test images...")

    for filename, text_content in sample_texts.items():
        try:
            # Create image
            width, height = 600, 800
            background_color = 'white'
            text_color = 'black'

            # Create different backgrounds to simulate different document types
            if 'classified' in filename:
                background_color = '#f8f8f8'  # Light gray for classified docs
            elif 'field' in filename:
                background_color = '#fffff0'  # Light yellow for field reports

            image = Image.new('RGB', (width, height), color=background_color)
            draw = ImageDraw.Draw(image)

            # Try to get a good font
            try:
                # Try common system fonts
                font_large = ImageFont.truetype("Arial.ttf", 18)
                font_medium = ImageFont.truetype("Arial.ttf", 14)
            except:
                try:
                    font_large = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 18)
                    font_medium = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 14)
                except:
                    # Fallback to default font
                    font_large = ImageFont.load_default()
                    font_medium = ImageFont.load_default()

            # Add border to simulate document
            border_color = '#cccccc'
            draw.rectangle([10, 10, width - 10, height - 10], outline=border_color, width=2)

            # Add header classification bar for classified docs
            if 'classified' in filename or 'threat' in filename:
                draw.rectangle([10, 10, width - 10, 50], fill='red', outline='darkred')
                draw.text((20, 20), "CLASSIFIED - SECRET", fill='white', font=font_large)
                text_start_y = 70
            else:
                text_start_y = 30

            # Draw the main text
            draw.multiline_text(
                (30, text_start_y),
                text_content,
                fill=text_color,
                font=font_medium,
                spacing=4
            )

            # Add some realistic document artifacts
            # Simulate slight rotation (common in scanned docs)
            if 'field' in filename:
                # Rotate slightly to simulate handheld scanning
                image = image.rotate(0.5, fillcolor=background_color)

            # Save image
            image_path = sample_dir / f"{filename}.png"
            image.save(image_path, 'PNG', quality=95)
            print(f"✅ Created: {image_path}")

            # Also create a JPEG version with slightly lower quality to simulate scanning
            jpeg_path = sample_dir / f"{filename}.jpg"
            image.save(jpeg_path, 'JPEG', quality=85)
            print(f"✅ Created: {jpeg_path}")

        except Exception as e:
            print(f"❌ Error creating {filename}: {e}")

    # Create a more challenging image (simulating poor scanning conditions)
    try:
        print("\n🎯 Creating challenging OCR test image...")

        # Low contrast, slightly blurry text
        width, height = 500, 300
        image = Image.new('RGB', (width, height), color='#f0f0f0')
        draw = ImageDraw.Draw(image)

        challenging_text = """INTERCEPTED COMMUNICATION
Source: Unknown
Target: EASTERN SYNDICATE
Amount: $847,000 USD
Location: PANAMA CITY
Contact: DMITRI VOLKOV
Status: Under investigation"""

        try:
            font = ImageFont.truetype("Arial.ttf", 12)
        except:
            font = ImageFont.load_default()

        # Draw text with lower contrast
        draw.multiline_text((20, 20), challenging_text, fill='#404040', font=font)

        # Add some noise to simulate poor scanning
        import random
        for _ in range(100):
            x, y = random.randint(0, width), random.randint(0, height)
            draw.point((x, y), fill='#e0e0e0')

        challenge_path = sample_dir / "challenging_scan.png"
        image.save(challenge_path, 'PNG')
        print(f"✅ Created challenging test: {challenge_path}")

    except Exception as e:
        print(f"❌ Error creating challenging image: {e}")

    print(f"\n🎉 OCR test images created in: {sample_dir}")
    print("📋 Test these images in your dashboard to verify OCR:")

    for file_path in sample_dir.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in ['.png', '.jpg']:
            file_size = file_path.stat().st_size
            print(f"   {file_path.name} ({file_size:,} bytes)")

    print("\n💡 OCR Testing Tips:")
    print("   • Upload these images in the dashboard")
    print("   • Check that OCR extracts the text content")
    print("   • Verify entity extraction works on OCR'd text")
    print("   • Try the challenging_scan.png for stress testing")


if __name__ == "__main__":
    create_sample_images()