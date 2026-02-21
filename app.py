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
            ai_output_pil = Image.open(io.BytesIO(part.inline_data.data))
            
            # [중요] 원본 A 이미지와 100% 동일한 해상도/비율로 강제 맞춤
            if ai_output_pil.size != img_a_pil.size:
                ai_output_pil = ai_output_pil.resize(img_a_pil.size, Image.Resampling.LANCZOS)
                
            return ai_output_pil
            
    raise ValueError("AI가 이미지를 반환하지 않았습니다.")

# --- UI 및 상태 관리 ---
st.title("🍌 Nano Banana Pro: AI 마킹 영역 패턴 자연 합성기")
st.markdown("💡 **파일 선택 방식:** 점선 박스에 **Drag & Drop** 하거나, 전용 버튼을 눌러 **Copy & Paste (클립보드)** 가 모두 가능합니다!")

api_key = st.sidebar.text_input("🔑 Google Gemini API Key 입력", type="password", key="input_api_key")

# Session State 초기화 (붙여넣기 상태 및 결과물 저장용)
if "pasted_a_image" not in st.session_state:
    st.session_state.pasted_a_image = None
if "pasted_b_images" not in st.session_state:
    st.session_state.pasted_b_images = {}
if "generated_results" not in st.session_state:
    st.session_state.generated_results = []

col1, col2 = st.columns(2)

with col1:
    st.subheader("1. 기준 이미지 (Image A)")
    file_a = st.file_uploader("📂 [Drag & Drop] 마킹된 원본 이미지", type=["png", "jpg", "jpeg"], key="uploader_img_a")
    
    st.markdown("또는 클립보드에 복사(Ctrl+C)한 후 아래 버튼 클릭:")
    paste_a_result = paste_image_button(
        label="📋 [Copy & Paste] 이미지 A 붙여넣기", 
        background_color="#4CAF50", 
        hover_background_color="#45a049", 
        key="paste_btn_a"
    )
    
    img_a_pil = None
    if file_a is not None:
        img_a_pil = Image.open(file_a).convert('RGB')
        st.session_state.pasted_a_image = None 
    elif paste_a_result.image_data is not None:
        img_a_pil = paste_a_result.image_data.convert('RGB')
        st.session_state.pasted_a_image = img_a_pil
    elif st.session_state.pasted_a_image is not None:
        img_a_pil = st.session_state.pasted_a_image

    if img_a_pil:
        st.image(img_a_pil, caption=f"✅ [준비 완료] 기준 이미지 A ({img_a_pil.width}x{img_a_pil.height})", use_container_width=True)

with col2:
    st.subheader("2. 패턴/분위기 이미지 (Image B들)")
    files_b = st.file_uploader("📂 [Drag & Drop] 패턴 이미지 (여러 장 가능)", type=["png", "jpg", "jpeg"], accept_multiple_files=True, key="uploader_img_b")
    
    st.markdown("또는 클립보드에 복사(Ctrl+C)한 후 계속해서 아래 버튼 클릭:")
    paste_b_result = paste_image_button(
        label="📋 [Copy & Paste] 패턴 이미지 B 붙여넣기", 
        background_color="#2196F3", 
        hover_background_color="#0b7dda", 
        key="paste_btn_b"
    )
    
    if paste_b_result.image_data is not None:
        img_hash = get_image_hash(paste_b_result.image_data)
        if img_hash not in st.session_state.pasted_b_images:
            st.session_state.pasted_b_images[img_hash] = paste_b_result.image_data.convert('RGB')

    all_b_images = []
    if files_b:
        for fb in files_b:
            all_b_images.append((fb.name, Image.open(fb).convert('RGB')))
            
    for i, (h, p_img) in enumerate(st.session_state.pasted_b_images.items()):
        all_b_images.append((f"pasted_image_{i+1}.jpg", p_img))

    if all_b_images:
        st.success(f"✅ 총 {len(all_b_images)}장의 패턴 이미지가 준비되었습니다.")
        with st.expander("🖼️ 준비된 패턴 이미지 미리보기 및 관리"):
            cols_b = st.columns(3)
            for idx, (b_name, b_img) in enumerate(all_b_images):
                cols_b[idx % 3].image(b_img, caption=b_name, use_container_width=True)
            
            if st.session_state.pasted_b_images:
                if st.button("🗑️ 붙여넣은 패턴 이미지 모두 지우기", key="btn_clear_b_images"):
                    st.session_state.pasted_b_images = {}
                    st.rerun()

st.divider()

# --- AI 처리 로직 ---
if img_a_pil and all_b_images:
    if st.button("🚀 AI 합성 시작하기", use_container_width=True, key="btn_start_ai_process"):
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
                        # 기존 결과물 초기화
                        st.session_state.generated_results = []
                        
                        for b_name, b_img in all_b_images:
                            # 100% 동일한 사이즈로 리사이즈된 결과물 획득
                            result_pil = process_with_nano_banana(api_key, img_a_pil, mask_np, b_img)
                            output_filename = f"ai_result_{b_name}"
                            
                            # Session State에 저장 (화면 새로고침 시 유지)
                            st.session_state.generated_results.append({
                                "filename": output_filename,
                                "image": result_pil
                            })
                            
                        st.success("🎉 AI 합성이 성공적으로 완료되었습니다! 아래에서 결과를 확인하세요.")
                except Exception as e:
                    st.error(f"API 호출 중 오류가 발생했습니다: {e}")

# --- 결과물 미리보기 및 선택적 다운로드 섹션 ---
if st.session_state.generated_results:
    st.header("🎯 생성된 결과물 선택 및 다운로드")
    st.info(f"모든 결과물은 원본 A 이미지와 동일한 크기({st.session_state.generated_results[0]['image'].width}x{st.session_state.generated_results[0]['image'].height})로 유지됩니다.")
    
    selected_images = []
    cols_res = st.columns(3)
    
    # 생성된 이미지들을 화면에 보여주고 체크박스 생성
    for idx, item in enumerate(st.session_state.generated_results):
        with cols_res[idx % 3]:
            st.image(item["image"], caption=item["filename"], use_container_width=True)
            # 체크박스 (기본값: True)
            is_selected = st.checkbox("이 이미지 다운로드", value=True, key=f"check_download_{idx}_{item['filename']}")
            
            if is_selected:
                selected_images.append(item)
    
    st.divider()
    
    # 선택된 이미지가 있을 때만 ZIP으로 묶어서 다운로드 제공
    if selected_images:
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for item in selected_images:
                img_byte_arr = io.BytesIO()
                item["image"].save(img_byte_arr, format='JPEG', quality=100)
                zip_file.writestr(item["filename"], img_byte_arr.getvalue())
        
        zip_buffer.seek(0)
        
        st.download_button(
            label=f"💾 선택한 이미지({len(selected_images)}장) 일괄 다운로드 (.zip)",
            data=zip_buffer,
            file_name="selected_nano_banana_results.zip",
            mime="application/zip",
            use_container_width=True,
            key="btn_download_selected_zip"
        )
    else:
        st.warning("다운로드할 이미지를 최소 1장 이상 선택해주세요.")
