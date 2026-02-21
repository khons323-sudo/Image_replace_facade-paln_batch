import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import zipfile
import hashlib
from google import genai

# 클립보드 붙여넣기를 완벽히 지원하는 Streamlit 확장 컴포넌트
from streamlit_paste_button import paste_image_button

# Streamlit 페이지 설정
st.set_page_config(page_title="AI 패턴 합성기 (Nano Banana Pro)", layout="wide")

def get_image_hash(pil_img):
    """이미지 중복 붙여넣기를 방지하기 위한 해시 생성 함수"""
    return hashlib.md5(pil_img.tobytes()).hexdigest()

def get_filled_red_mask(image_np):
    """이미지에서 빨간색 테두리를 찾고 안쪽 영역까지 채운 마스크 반환"""
    hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
    
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    
    red_border_mask = mask1 + mask2
    kernel = np.ones((5,5), np.uint8)
    red_border_mask = cv2.morphologyEx(red_border_mask, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(red_border_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled_mask = np.zeros_like(red_border_mask)
    cv2.drawContours(filled_mask, contours, -1, (255), thickness=cv2.FILLED)
    
    return cv2.bitwise_or(filled_mask, red_border_mask)

def process_with_nano_banana(api_key, img_a_pil, mask_np, img_b_pil):
    """나노 바나나 프로(Gemini) API를 호출하여 이미지를 합성"""
    client = genai.Client(api_key=api_key)
    mask_pil = Image.fromarray(mask_np).convert("L")
    
    prompt = """
    You are an expert AI image editor.
    I have provided three images in order:
    1. Base Image (contains red marked lines)
    2. Mask Image (white area indicates the inside of the red marking and the marking itself)
    3. Reference Style Image
    
    Task: 
    1. Remove the red marking lines completely from the Base Image.
    2. Inpaint the area indicated by the Mask Image naturally using the pattern and atmosphere of the Reference Style Image.
    3. Ensure the boundaries are seamlessly blended and lighting matches.
    Output ONLY the seamlessly edited image.
    """
    
    response = client.models.generate_content(
        model='gemini-3-pro-image-preview',
        contents=[prompt, img_a_pil, mask_pil, img_b_pil]
    )
    
    for part in response.candidates[0].content.parts:
        if part.inline_data:
            return Image.open(io.BytesIO(part.inline_data.data))
            
    raise ValueError("AI가 이미지를 반환하지 않았습니다.")

# --- UI 및 상태 관리 ---
st.title("🍌 Nano Banana Pro: AI 마킹 영역 패턴 자연 합성기")
st.markdown("💡 **파일 선택 방식:** 점선 박스에 **Drag & Drop** 하거나, 전용 버튼을 눌러 **Copy & Paste (클립보드)** 가 모두 가능합니다!")

api_key = st.sidebar.text_input("🔑 Google Gemini API Key 입력", type="password")

# 클립보드 붙여넣기 이미지들을 저장할 Session State 초기화
if "pasted_a_image" not in st.session_state:
    st.session_state.pasted_a_image = None
if "pasted_b_images" not in st.session_state:
    st.session_state.pasted_b_images = {}

col1, col2 = st.columns(2)

with col1:
    st.subheader("1. 기준 이미지 (Image A)")
    
    # 1. Drag & Drop 영역
    file_a = st.file_uploader("📂 [Drag & Drop] 마킹된 원본 이미지", type=["png", "jpg", "jpeg"], key="img_a")
    
    # 2. Copy & Paste 영역
    st.markdown("또는 클립보드에 복사(Ctrl+C)한 후 아래 버튼 클릭:")
    paste_a_result = paste_image_button(label="📋 [Copy & Paste] 이미지 A 붙여넣기", background_color="#4CAF50", hover_background_color="#45a049")
    
    # 두 소스(드래그, 붙여넣기) 중 하나라도 있으면 img_a_pil로 설정
    img_a_pil = None
    if file_a is not None:
        img_a_pil = Image.open(file_a).convert('RGB')
        st.session_state.pasted_a_image = None  # 파일이 우선시되도록 기존 붙여넣기 초기화
    elif paste_a_result.image_data is not None:
        img_a_pil = paste_a_result.image_data.convert('RGB')
        st.session_state.pasted_a_image = img_a_pil
    elif st.session_state.pasted_a_image is not None:
        img_a_pil = st.session_state.pasted_a_image

    if img_a_pil:
        st.image(img_a_pil, caption="✅ [준비 완료] 기준 이미지 A", use_container_width=True)

with col2:
    st.subheader("2. 패턴/분위기 이미지 (Image B들)")
    
    # 1. Drag & Drop 영역
    files_b = st.file_uploader("📂 [Drag & Drop] 패턴 이미지 (여러 장 가능)", type=["png", "jpg", "jpeg"], accept_multiple_files=True, key="img_b")
    
    # 2. Copy & Paste 영역
    st.markdown("또는 클립보드에 복사(Ctrl+C)한 후 계속해서 아래 버튼 클릭:")
    paste_b_result = paste_image_button(label="📋 [Copy & Paste] 패턴 이미지 B 붙여넣기", background_color="#2196F3", hover_background_color="#0b7dda")
    
    # 붙여넣은 B 이미지는 리스트(Session State)에 누적 저장
    if paste_b_result.image_data is not None:
        img_hash = get_image_hash(paste_b_result.image_data)
        if img_hash not in st.session_state.pasted_b_images:
            st.session_state.pasted_b_images[img_hash] = paste_b_result.image_data.convert('RGB')

    # Drag & Drop 된 파일 + Copy & Paste 된 파일 하나로 합치기
    all_b_images = []
    if files_b:
        for fb in files_b:
            all_b_images.append((fb.name, Image.open(fb).convert('RGB')))
            
    for i, (h, p_img) in enumerate(st.session_state.pasted_b_images.items()):
        all_b_images.append((f"pasted_image_{i+1}.jpg", p_img))

    # 취합된 B 이미지 상태 표시 및 관리
    if all_b_images:
        st.success(f"✅ 총 {len(all_b_images)}장의 패턴 이미지가 준비되었습니다.")
        with st.expander("🖼️ 준비된 패턴 이미지 미리보기 및 관리"):
            cols = st.columns(3)
            for idx, (b_name, b_img) in enumerate(all_b_images):
                cols[idx % 3].image(b_img, caption=b_name, use_container_width=True)
            
            # 붙여넣은 이미지 초기화 버튼
            if st.session_state.pasted_b_images:
                if st.button("🗑️ 붙여넣은 패턴 이미지 모두 지우기"):
                    st.session_state.pasted_b_images = {}
                    st.rerun()

st.divider()

# --- AI 처리 및 저장 로직 ---
if img_a_pil and all_b_images:
    if st.button("🚀 AI 자동 합성 및 일괄 다운로드 준비", use_container_width=True):
        if not api_key:
            st.error("좌측 사이드바에 Google Gemini API Key를 입력해주세요!")
        else:
            with st.spinner("🍌 나노 바나나 프로 AI가 빛과 질감을 살려 자연스럽게 합성 중입니다... (1장 당 수 초 소요)"):
                try:
                    img_a_np = np.array(img_a_pil)
                    mask_np = get_filled_red_mask(img_a_np)
                    
                    if cv2.countNonZero(mask_np) == 0:
                        st.error("기준 이미지에서 빨간색 마킹을 찾을 수 없습니다.")
                    else:
                        zip_buffer = io.BytesIO()
                        
                        # 취합된 전체 B 이미지를 순회하며 AI 합성
                        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                            for b_name, b_img in all_b_images:
                                # AI 합성 실행
                                result_pil = process_with_nano_banana(api_key, img_a_pil, mask_np, b_img)
                                
                                # 메모리에 압축
                                img_byte_arr = io.BytesIO()
                                result_pil.save(img_byte_arr, format='JPEG', quality=95)
                                
                                # 고유 파일명 지정
                                output_filename = f"ai_result_{b_name}"
                                zip_file.writestr(output_filename, img_byte_arr.getvalue())
                        
                        zip_buffer.seek(0)
                        st.success("🎉 AI 합성이 성공적으로 완료되었습니다!")
                        
                        # 일괄 다운로드
                        st.download_button(
                            label="💾 전체 결과 이미지 일괄 다운로드 (.zip)",
                            data=zip_buffer,
                            file_name="nano_banana_results.zip",
                            mime="application/zip",
                            use_container_width=True
                        )
                except Exception as e:
                    st.error(f"API 호출 중 오류가 발생했습니다: {e}")
