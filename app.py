import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import zipfile

# Streamlit 페이지 설정
st.set_page_config(page_title="이미지 패턴 합성기", layout="wide")

def get_red_mask(image_np):
    """이미지에서 빨간색 영역을 찾아 마스크로 반환합니다."""
    hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
    
    # 빨간색은 HSV 색상 공간에서 양끝(0 근처, 180 근처)에 분포합니다.
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    
    mask = mask1 + mask2
    
    # 노이즈 제거를 위해 모폴로지 연산 적용
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

def process_image(img_a_np, img_b_np, mask):
    """A 이미지의 마스크 영역에 B 이미지의 패턴을 합성합니다."""
    # 1. 빨간색 마킹 선 지우기 (주변 색상으로 채움)
    inpainted_a = cv2.inpaint(img_a_np, mask, 3, cv2.INPAINT_TELEA)
    
    # 2. B 이미지를 A 이미지 크기에 맞게 리사이즈
    img_b_resized = cv2.resize(img_b_np, (img_a_np.shape[1], img_a_np.shape[0]))
    
    # 3. 경계를 부드럽게 합성하기 위해 마스크 블러 처리
    mask_blurred = cv2.GaussianBlur(mask, (15, 15), 0)
    mask_float = mask_blurred.astype(float) / 255.0
    mask_3d = np.repeat(mask_float[:, :, np.newaxis], 3, axis=2)
    
    # 4. 합성: 마스크 영역 밖은 지워진 A이미지, 안쪽은 B이미지 패턴
    blended = (inpainted_a * (1 - mask_3d) + img_b_resized * mask_3d).astype(np.uint8)
    
    return blended

# --- UI 구현 ---
st.title("🎨 빨간펜 영역 패턴/분위기 일괄 합성 프로그램")
st.markdown("빨간선으로 마킹된 기준 이미지(A)와 패턴으로 사용할 이미지들(B1, B2...)을 업로드하세요. 드래그 앤 드롭 및 파일 탐색기 창에서의 복사&붙여넣기를 지원합니다.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("1. 기준 이미지 (Image A) 업로드")
    file_a = st.file_uploader("빨간선이 마킹된 이미지를 업로드하세요.", type=["png", "jpg", "jpeg"], key="img_a")

with col2:
    st.subheader("2. 패턴 이미지 (Image B들) 업로드")
    files_b = st.file_uploader("패턴/분위기를 가져올 이미지들을 여러 장 선택하세요.", type=["png", "jpg", "jpeg"], accept_multiple_files=True, key="img_b")

if file_a and files_b:
    st.success(f"기준 이미지 1장과 패턴 이미지 {len(files_b)}장이 준비되었습니다.")
    
    if st.button("🚀 일괄 합성 및 결과 생성", use_container_width=True):
        with st.spinner("이미지 합성 중..."):
            # A 이미지 로드 및 마스크 추출
            img_a_pil = Image.open(file_a).convert('RGB')
            img_a_np = np.array(img_a_pil)
            mask = get_red_mask(img_a_np)
            
            # 빨간색이 검출되지 않았을 경우 예외 처리
            if cv2.countNonZero(mask) == 0:
                st.error("기준 이미지에서 빨간색 마킹을 찾을 수 없습니다. 색상이나 이미지를 확인해주세요.")
            else:
                # 결과를 저장할 ZIP 파일 메모리 버퍼
                zip_buffer = io.BytesIO()
                
                with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                    for idx, file_b in enumerate(files_b):
                        img_b_pil = Image.open(file_b).convert('RGB')
                        img_b_np = np.array(img_b_pil)
                        
                        # 합성 처리
                        result_np = process_image(img_a_np, img_b_np, mask)
                        result_pil = Image.fromarray(result_np)
                        
                        # 이미지를 메모리에 저장
                        img_byte_arr = io.BytesIO()
                        result_pil.save(img_byte_arr, format='JPEG')
                        
                        # 원본 B 파일명에 기반하여 결과 파일명 생성
                        output_filename = f"result_{file_b.name}"
                        zip_file.writestr(output_filename, img_byte_arr.getvalue())
                
                # ZIP 파일 준비 완료
                zip_buffer.seek(0)
                
                st.success("✅ 합성이 완료되었습니다! 아래 버튼을 눌러 저장할 위치를 선택하세요.")
                
                # 저장 위치 선택(다운로드) 버튼
                st.download_button(
                    label="💾 전체 결과 이미지 일괄 다운로드 (.zip)",
                    data=zip_buffer,
                    file_name="processed_images_result.zip",
                    mime="application/zip",
                    use_container_width=True
                )
