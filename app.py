import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import zipfile
from google import genai
from google.genai import types

# Streamlit 페이지 설정
st.set_page_config(page_title="AI 패턴 합성기 (Nano Banana Pro)", layout="wide")

def get_filled_red_mask(image_np):
    """이미지에서 빨간색 테두리를 찾고, 그 안쪽 영역까지 꽉 채운 마스크를 반환합니다."""
    hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
    
    # 빨간색 추출 (HSV 공간)
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    
    red_border_mask = mask1 + mask2
    
    # 노이즈 제거
    kernel = np.ones((5,5), np.uint8)
    red_border_mask = cv2.morphologyEx(red_border_mask, cv2.MORPH_CLOSE, kernel)
    
    # 빨간선 안쪽 영역 채우기 (컨투어 추출)
    contours, _ = cv2.findContours(red_border_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled_mask = np.zeros_like(red_border_mask)
    
    # 찾은 빨간선 내부를 하얗게 채움 (AI가 인식할 인페인팅 대상 영역)
    cv2.drawContours(filled_mask, contours, -1, (255), thickness=cv2.FILLED)
    
    # 원본 빨간선 자체도 삭제/수정하기 위해 마스크 병합
    final_mask = cv2.bitwise_or(filled_mask, red_border_mask)
    return final_mask

def process_with_nano_banana(api_key, img_a_pil, mask_np, img_b_pil):
    """나노 바나나 프로(Gemini 3 Pro Image) API를 호출하여 이미지를 합성합니다."""
    # Google GenAI 클라이언트 초기화
    client = genai.Client(api_key=api_key)
    
    # Numpy 마스크를 PIL 이미지로 변환
    mask_pil = Image.fromarray(mask_np).convert("L")
    
    # AI에게 내릴 멀티모달 프롬프트 지시어
    prompt = """
    You are an expert AI image editor.
    I have provided three images in order:
    1. Base Image (contains red marked lines)
    2. Mask Image (white area indicates the inside of the red marking and the marking itself)
    3. Reference Style Image
    
    Task: 
    1. Remove the red marking lines completely from the Base Image.
    2. Inpaint the area indicated by the Mask Image. Fill this area naturally using the pattern, texture, and atmosphere of the Reference Style Image.
    3. Ensure the boundaries are seamlessly blended and the lighting/shadows match the rest of the Base Image.
    Output ONLY the seamlessly edited image.
    """
    
    # Nano Banana Pro (gemini-3-pro-image-preview) 모델 호출
    response = client.models.generate_content(
        model='gemini-3-pro-image-preview',
        contents=[
            prompt, 
            img_a_pil, 
            mask_pil, 
            img_b_pil
        ]
    )
    
    # AI가 생성한 이미지 결과물 추출
    for part in response.candidates[0].content.parts:
        if part.inline_data:
            return Image.open(io.BytesIO(part.inline_data.data))
            
    raise ValueError("AI가 이미지를 반환하지 않았습니다. 프롬프트나 마스크를 확인해주세요.")

# --- UI 구현 ---
st.title("🍌 Nano Banana Pro: AI 마킹 영역 패턴 자연 합성기")
st.markdown("""
**나노바나나프로(Gemini 3 Pro Image)** API를 활용해 빨간선 안쪽을 자연스럽게 채워줍니다.
* 💡 **파일 업로드 팁:** 점선 박스 안에 파일을 **드래그 앤 드롭** 하거나, 박스를 한 번 클릭한 후 **`Ctrl + V` (붙여넣기)** 하시면 클립보드 이미지가 바로 업로드됩니다!
""")

# API 키 입력 (보안을 위해 비밀번호 형태로 마스킹)
api_key = st.sidebar.text_input("🔑 Google Gemini API Key 입력", type="password", help="Google AI Studio에서 발급받은 API 키를 입력하세요.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("1. 기준 이미지 (Image A)")
    file_a = st.file_uploader("빨간선이 마킹된 원본 이미지를 업로드하세요.", type=["png", "jpg", "jpeg"], key="img_a")

with col2:
    st.subheader("2. 패턴/분위기 이미지 (Image B들)")
    files_b = st.file_uploader("안쪽을 채울 패턴 이미지들을 선택하세요. (여러 장 가능)", type=["png", "jpg", "jpeg"], accept_multiple_files=True, key="img_b")

if file_a and files_b:
    st.success(f"기준 이미지 1장과 패턴 이미지 {len(files_b)}장이 준비되었습니다.")
    
    if st.button("🚀 AI 자동 합성 및 일괄 다운로드 준비", use_container_width=True):
        if not api_key:
            st.error("좌측 사이드바에 Google Gemini API Key를 입력해주세요!")
        else:
            with st.spinner("🍌 나노 바나나 프로 AI가 이미지를 자연스럽게 합성 중입니다... (시간이 조금 걸릴 수 있습니다)"):
                try:
                    # A 이미지 로드 및 AI용 마스크 추출
                    img_a_pil = Image.open(file_a).convert('RGB')
                    img_a_np = np.array(img_a_pil)
                    mask_np = get_filled_red_mask(img_a_np)
                    
                    if cv2.countNonZero(mask_np) == 0:
                        st.error("기준 이미지에서 빨간색 마킹을 찾을 수 없습니다.")
                    else:
                        zip_buffer = io.BytesIO()
                        
                        # 다중 이미지 일괄 처리
                        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                            for idx, file_b in enumerate(files_b):
                                img_b_pil = Image.open(file_b).convert('RGB')
                                
                                # Nano Banana Pro API 호출
                                result_pil = process_with_nano_banana(api_key, img_a_pil, mask_np, img_b_pil)
                                
                                # 메모리에 이미지 저장
                                img_byte_arr = io.BytesIO()
                                result_pil.save(img_byte_arr, format='JPEG', quality=95)
                                
                                output_filename = f"ai_result_{file_b.name}"
                                zip_file.writestr(output_filename, img_byte_arr.getvalue())
                        
                        zip_buffer.seek(0)
                        st.success("✅ AI 합성이 성공적으로 완료되었습니다!")
                        
                        # 일괄 다운로드 버튼
                        st.download_button(
                            label="💾 전체 결과 이미지 일괄 다운로드 (.zip)",
                            data=zip_buffer,
                            file_name="nano_banana_results.zip",
                            mime="application/zip",
                            use_container_width=True
                        )
                except Exception as e:
                    st.error(f"API 호출 중 오류가 발생했습니다: {e}")
