import streamlit as st
import cv2
import numpy as np
from gtts import gTTS
import tempfile
import os
from PIL import Image
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Braille mapping dictionary
BRAILLE_MAP = {
    '100000': 'a', '110000': 'b', '100100': 'c', '100110': 'd', '100010': 'e',
    '110100': 'f', '110110': 'g', '110010': 'h', '010100': 'i', '010110': 'j',
    '101000': 'k', '111000': 'l', '101100': 'm', '101110': 'n', '101010': 'o',
    '111100': 'p', '111110': 'q', '111010': 'r', '011100': 's', '011110': 't',
    '101001': 'u', '111001': 'v', '010111': 'w', '101101': 'x', '101111': 'y',
    '101011': 'z', '000000': ' ', '001111': '.', '000011': ',', '000101': '?'
}

def preprocess_image(uploaded_file):
    """Convert uploaded file to OpenCV image with preprocessing"""
    try:
        image = Image.open(uploaded_file)
        img_array = np.array(image.convert('RGB'))
        img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        return thresh
    except Exception as e:
        logger.error(f"Image processing failed: {str(e)}")
        return None

def detect_braille_dots(image):
    """Detect braille dots using contour analysis"""
    try:
        contours, _ = cv2.findContours(
            image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        MIN_DOT_AREA = 5
        MAX_DOT_AREA = 100
        dots = []
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if MIN_DOT_AREA < area < MAX_DOT_AREA:
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    dots.append((cx, cy))
        return dots
    except Exception as e:
        logger.error(f"Dot detection failed: {str(e)}")
        return []

def group_dots_into_lines(dots, y_tolerance=20):
    """Group dots into lines based on y-coordinate"""
    if not dots:
        return []
    
    try:
        y_coords = sorted({dot[1] for dot in dots})
        lines = []
        current_line = [y_coords[0]]
        
        for y in y_coords[1:]:
            if y - current_line[-1] < y_tolerance:
                current_line.append(y)
            else:
                lines.append(current_line)
                current_line = [y]
        lines.append(current_line)
        
        grouped_lines = []
        for line in lines:
            line_y = sum(line) // len(line)
            line_dots = [dot for dot in dots if abs(dot[1] - line_y) < y_tolerance]
            grouped_lines.append(sorted(line_dots, key=lambda x: x[0]))
        return grouped_lines
    except Exception as e:
        logger.error(f"Line grouping failed: {str(e)}")
        return []

def convert_to_braille_pattern(line_dots, char_spacing=40):
    """Convert dot positions to 6-bit braille patterns"""
    if not line_dots:
        return ""
    
    try:
        # Group dots into characters
        characters = []
        current_char = [line_dots[0]]
        
        for dot in line_dots[1:]:
            if dot[0] - current_char[-1][0] < char_spacing:
                current_char.append(dot)
            else:
                characters.append(current_char)
                current_char = [dot]
        characters.append(current_char)
        
        # Create braille patterns for each character
        line_text = ""
        for char_dots in characters:
            pattern = ['0']*6
            
            if char_dots:
                # Calculate character bounding box
                xs = [dot[0] for dot in char_dots]
                ys = [dot[1] for dot in char_dots]
                min_x, max_x = min(xs), max(xs)
                min_y, max_y = min(ys), max(ys)
                width = max_x - min_x
                height = max_y - min_y
                
                # Assign dot positions
                for x, y in char_dots:
                    col = 0 if x < min_x + width/2 else 1
                    row = 0 if y < min_y + height/3 else (1 if y < min_y + 2*height/3 else 2)
                    pos = row + 3*col
                    if pos < 6:
                        pattern[pos] = '1'
            
            pattern_str = ''.join(pattern)
            line_text += BRAILLE_MAP.get(pattern_str, '?')
        return line_text
    except Exception as e:
        logger.error(f"Pattern conversion failed: {str(e)}")
        return ""

def text_to_speech(text, lang='en'):
    """Convert text to speech and return audio file path"""
    if not text.strip():
        return None
    try:
        tts = gTTS(text=text, lang=lang, slow=False)
        audio_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
        tts.save(audio_file.name)
        return audio_file.name
    except Exception as e:
        logger.error(f"Text-to-speech failed: {str(e)}")
        return None

# Streamlit UI
st.title("Braille to Text & Speech Converter")
st.markdown("Upload an image of Braille text to convert to spoken English")

uploaded_file = st.file_uploader("Choose a Braille image:", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    # Image processing
    processed_img = preprocess_image(uploaded_file)
    
    if processed_img is None:
        st.error("Failed to process image. Please try another file.")
    else:
        dots = detect_braille_dots(processed_img)
        
        if not dots:
            st.warning("No Braille dots detected. Please check image quality.")
        else:
            lines = group_dots_into_lines(dots)
            output_text = "\n".join(convert_to_braille_pattern(line) for line in lines)
            
            st.subheader("Converted Text:")
            st.code(output_text)
            
            # Text-to-speech
            audio_path = text_to_speech(output_text)
            if audio_path:
                st.audio(audio_path, format='audio/mp3')
                try:
                    os.unlink(audio_path)  # Clean up temp file
                except:
                    pass

# Add image processing tips
st.sidebar.markdown("**Image Tips:**")
st.sidebar.write("- Use high-contrast images")
st.sidebar.write("- Ensure dots are clearly visible")
st.sidebar.write("- Ideal resolution: 300+ DPI")
st.sidebar.write("- Avoid shadows and glare")
