import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import zipfile
import hashlib
import base64
from google import genai

# --- [중요 패치] Streamlit 1.40.0+ 호환성 영구 해결 (Base64 인코딩 우회) ---
import streamlit.elements.image as st_image
def custom_image_to_url(image, width=None, clamp=False, channels="RGB", output_format="PNG", image_id="", *args, **kwargs):
    """Streamlit 내부 API를 타지 않고 이미지를 Base64 Data URI로 직접 변환"""
    try:
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        buffered = io.BytesIO()
        # RGBA 모드일 경우 포맷 충돌 방지
        if image.mode == "RGBA" and output_format.upper() == "JPEG":
            image = image.convert("RGB")
            
        fmt = output_format if output_format else "PNG"
        image.save(buffered, format=fmt)
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/{fmt.lower()};base64,{img_str}"
    except Exception as e:
        st.error(f"Image to URL 변환 에러: {e}")
        return ""

# st_canvas가 호출하는 구버전 함수를 커스텀 함수로 완벽히 덮어씌움
st_image.image_to_url = custom_image_to_url
# -------------------------------------------------------------------------

# 클립보드 붙여넣기 컴포넌트
from streamlit_paste_button import paste_image_button
# 직접 마킹(그리기) 지원 컴포넌트
from streamlit_drawable_canvas import st_canvas

# Streamlit 페이지 설정
st.set_page_config(page_title="AI 패턴 합성기 (Nano Banana Pro)", layout="wide")

def get_image_hash(pil_img):
    """이미지 중복 방지를 위한 해시 생성"""
    return hashlib.md5(pil_img.tobytes()).hexdigest()

def get_mask_from_canvas(canvas_image_data):
    """캔버스 데이터(RGBA)에서 사용자가 그린 부분과 안쪽을 꽉 채운 마스크 추출"""
    if canvas_image_data is None:
        return None
    
    alpha = canvas_image_data[:, :, 3]
    drawn_mask = (alpha > 0).astype(np.uint8) * 255
    
    kernel = np.ones((5,5), np.uint8)
    drawn_mask = cv2.morphologyEx(drawn_mask, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(drawn_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled_mask = np.zeros_like(drawn_mask)
    cv2.drawContours(filled_mask, contours, -1, (255), thickness=cv2.FILLED)
    
    final_mask = cv2.bitwise_or(filled_mask, drawn_mask)
    return final_mask

def strict_composite(original_img_np, generated_img_np, mask_np):
    """마킹되지 않은 원본 영역 100% 보존, 마킹된 부분만 AI 이미지로 교체"""
    h, w = original_img_np.shape[:2]
    generated_resized = cv2.resize(generated_img_np, (w, h))
    mask_3d = np.repeat(mask_np[:, :, np.newaxis], 3, axis=2)
    final_img_np = np.where(mask_3d > 0, generated_resized, original_img_np)
    return final_img_np

def process_with_nano_banana(api_key, img_a_pil, mask_np, img_b_pil):
    """나노 바나나 프로(Gemini) API 호출"""
    client = genai.Client(api_key=api_key)
    mask_pil = Image.fromarray(mask_np).convert("L")
    
    prompt = """
    You are an expert AI image editor.
    1. Base Image
    2. Mask Image (white area is the target)
    3. Reference Style Image
    
    Task: Inpaint the masked area ONLY naturally using the pattern, texture, and atmosphere of the Reference Style Image. Output ONLY the edited image.
    """
    response = client.models.generate_content(
        model='gemini-3-pro-image-preview',
        contents=[prompt, img_a_pil, mask_pil, img_b_pil]
    )
    
    for part in response.candidates[0].content.parts:
        if part.inline_data:
            return Image.open(io.BytesIO(part.inline_data.data)).convert('RGB')
    raise ValueError("AI가 이미지를 반환하지 않았습니다.")

# --- 세션 상태 초기화 ---
if "pasted_a_image" not in st.session_state:
    st.session_state.pasted_a_image = None
if "pasted_b_images" not in st.session_state:
    st.session_state.pasted_b_images = {}
if "generated_results" not in st.session_state:
    st.session_state.generated_results = []

# --- UI 구현 ---
st.title("🍌 Nano Banana Pro: AI 마킹 영역 패턴 자연 합성기")
st.markdown("💡 **진행 순서:** 기준 이미지 업로드 ➡️ 직접 마킹 ➡️ 패턴 이미지 업로드 ➡️ AI 합성 ➡️ 결과 확인 및 선택 다운로드")

api_key = st.sidebar.text_input("🔑 Google Gemini API Key 입력", type="password", key="input_api_key")

st.header("Step 1. 기준 이미지 (Image A) 업로드 및 마킹")
col_a1, col_a2 = st.columns([1, 2])

with col_a1:
    file_a = st.file_uploader("📂 [Drag & Drop] 기준 이미지", type=["png", "jpg", "jpeg"], key="uploader_img_a")
    st.markdown("또는 클립보드 붙여넣기(Ctrl+C 후 클릭):")
    paste_a_result = paste_image_button(
        label="📋 [Copy & Paste] 이미지 A 붙여넣기", 
        background_color="#4CAF50", hover_background_color="#45a049", key="paste_btn_a"
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

with col_a2:
    if img_a_pil:
        st.subheader("🖍️ 이미지 마킹 (빨간색으로 적용할 영역 그리기)")
        st.markdown("왼쪽 하단의 🗑️(휴지통) 또는 ↩️(실행취소) 버튼을 눌러 그리기 취소가 가능합니다.")
        
        drawing_mode_kr = st.radio("도구 선택:", ["자유곡선 (자유롭게 그리기)", "직선 (선 긋기)", "원형 (동그라미)"], horizontal=True, key="tool_select")
        mode_map = {"자유곡선 (자유롭게 그리기)": "freedraw", "직선 (선 긋기)": "line", "원형 (동그라미)": "circle"}
        drawing_mode = mode_map[drawing_mode_kr]
        
        stroke_width = st.slider("펜 굵기", 1, 50, 15, key="stroke_width")
        
        # 캔버스 크기 최적화
        max_width = 800
        canvas_w, canvas_h = img_a_pil.width, img_a_pil.height
        if canvas_w > max_width:
            ratio = max_width / canvas_w
            canvas_w = max_width
            canvas_h = int(canvas_h * ratio)
            
        img_a_resized_for_canvas = img_a_pil.resize((canvas_w, canvas_h))

        # 캔버스 컴포넌트 렌더링
        canvas_result = st_canvas(
            fill_color="rgba(255, 0, 0, 0.3)", 
            stroke_width=stroke_width,
            stroke_color="#FF0000",             
            background_image=img_a_resized_for_canvas,
            update_streamlit=True,
            height=canvas_h,
            width=canvas_w,
            drawing_mode=drawing_mode,
            key="canvas",
        )

st.divider()

st.header("Step 2. 패턴/분위기 이미지 (Image B) 업로드")
col_b1, col_b2 = st.columns([1, 2])

with col_b1:
    files_b = st.file_uploader("📂 [Drag & Drop] 패턴 이미지 (여러 장 가능)", type=["png", "jpg", "jpeg"], accept_multiple_files=True, key="uploader_img_b")
    st.markdown("또는 클립보드 붙여넣기(연속 가능):")
    paste_b_result = paste_image_button(
        label="📋 [Copy & Paste] 패턴 이미지 붙여넣기", 
        background_color="#2196F3", hover_background_color="#0b7dda", key="paste_btn_b"
    )
    
    if paste_b_result.image_data is not None:
        img_hash = get_image_hash(paste_b_result.image_data)
        if img_hash not in st.session_state.pasted_b_images:
            st.session_state.pasted_b_images[img_hash] = paste_b_result.image_data.convert('RGB')

with col_b2:
    all_b_images = []
    if files_b:
        for fb in files_b:
            all_b_images.append((fb.name, Image.open(fb).convert('RGB')))
    for i, (h, p_img) in enumerate(st.session_state.pasted_b_images.items()):
        all_b_images.append((f"pasted_image_{i+1}.jpg", p_img))

    if all_b_images:
        st.success(f"✅ 총 {len(all_b_images)}장의 패턴 이미지가 준비되었습니다.")
        with st.expander("🖼️ 준비된 패턴 이미지 미리보기"):
            cols = st.columns(3)
            for idx, (b_name, b_img) in enumerate(all_b_images):
                cols[idx % 3].image(b_img, caption=b_name, use_container_width=True)
            if st.session_state.pasted_b_images:
                if st.button("🗑️ 붙여넣은 이미지 모두 지우기", key="btn_clear_b"):
                    st.session_state.pasted_b_images = {}
                    st.rerun()

st.divider()

st.header("Step 3. AI 자동 합성")
if img_a_pil and all_b_images:
    if st.button("🚀 선택한 영역에 패턴 합성 실행", use_container_width=True, key="btn_start_ai"):
        if not api_key:
            st.error("좌측 사이드바에 Google Gemini API Key를 입력해주세요!")
        elif canvas_result.image_data is None:
            st.error("이미지에 영역을 마킹(그리기) 해주세요.")
        else:
            with st.spinner("🍌 나노 바나나 프로 AI 합성 중... (원본 이미지 형태 완벽 보존 처리 중)"):
                try:
                    mask_np_resized = get_mask_from_canvas(canvas_result.image_data)
                    mask_np = cv2.resize(mask_np_resized, (img_a_pil.width, img_a_pil.height), interpolation=cv2.INTER_NEAREST)
                    
                    if cv2.countNonZero(mask_np) == 0:
                        st.error("그려진 마킹 영역이 없습니다. Step 1에서 캔버스에 빨간 펜으로 영역을 그려주세요.")
                    else:
                        img_a_np = np.array(img_a_pil)
                        results_temp = []
                        
                        for b_name, b_img in all_b_images:
                            ai_output_pil = process_with_nano_banana(api_key, img_a_pil, mask_np, b_img)
                            ai_output_np = np.array(ai_output_pil)
                            
                            final_np = strict_composite(img_a_np, ai_output_np, mask_np)
                            final_pil = Image.fromarray(final_np)
                            
                            results_temp.append({
                                "name": f"result_{b_name}",
                                "image": final_pil
                            })
                            
                        st.session_state.generated_results = results_temp
                        st.success("🎉 합성이 완료되었습니다! 아래에서 결과를 확인하세요.")
                except Exception as e:
                    st.error(f"처리 중 오류 발생: {e}")

st.divider()

if st.session_state.generated_results:
    st.header("Step 4. 결과 확인 및 다운로드")
    
    selected_files = []
    cols = st.columns(3)
    
    for idx, res in enumerate(st.session_state.generated_results):
        with cols[idx % 3]:
            st.image(res["image"], caption=res["name"], use_container_width=True)
            if st.checkbox(f"저장 선택: {res['name']}", value=True, key=f"chk_{res['name']}_{idx}"):
                selected_files.append(res)
                
    if selected_files:
        st.write(f"선택된 파일 수: **{len(selected_files)}**장")
        
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for item in selected_files:
                img_byte_arr = io.BytesIO()
                item["image"].save(img_byte_arr, format='JPEG', quality=100)
                zip_file.writestr(item["name"], img_byte_arr.getvalue())
        
        zip_buffer.seek(0)
        
        st.download_button(
            label="💾 선택한 이미지 일괄 다운로드 (.zip)",
            data=zip_buffer,
            file_name="selected_banana_results.zip",
            mime="application/zip",
            use_container_width=True,
            key="btn_download_selected_zip"
        )
