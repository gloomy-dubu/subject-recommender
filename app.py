import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# ==========================================
# 0. 페이지 기본 설정
# ==========================================
st.set_page_config(
    page_title="과목추천 AI",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 우리 학교 과목추천 AI")
st.markdown("### 진로에 딱 맞는 선택 과목을 찾아보세요!")
st.divider()

# ==========================================
# 1. 데이터 로드 (캐싱 적용으로 속도 향상)
# ==========================================
@st.cache_data
def load_data():
    try:
        # 파일명은 실제 업로드할 파일명과 일치해야 합니다.
        school_df = pd.read_excel('school_subjects.xlsx')
        univ_df = pd.read_excel('univ_req1.xlsx')
        
        # 전처리
        if '관련키워드' not in univ_df.columns: 
            univ_df['관련키워드'] = ''
        univ_df['관련키워드'] = univ_df['관련키워드'].fillna('')
        univ_df['학과명'] = univ_df['학과명'].fillna('')
        univ_df['search_text'] = univ_df['학과명'] + " " + univ_df['관련키워드'].astype(str)
        
        return school_df, univ_df
    except Exception as e:
        return None, None

school_df, univ_df = load_data()

if school_df is None:
    st.error("데이터 파일을 찾을 수 없습니다. (school_subjects.xlsx, univ_req1.xlsx)")
    st.stop()

# ==========================================
# 2. 로직 함수들 (유사도 분석 & 정규화)
# ==========================================
def normalize(text):
    if pd.isna(text): return ""
    return str(text).replace(" ", "").lower().strip()

def find_best_major_smart(user_input, univ_df):
    # 1. 포함 여부 확인
    mask = univ_df['search_text'].str.contains(user_input, case=False, na=False)
    matched_df = univ_df[mask]
    if not matched_df.empty:
        return matched_df.iloc[0], "match"

    # 2. 유사도 분석
    try:
        tfidf = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 3))
        documents = univ_df['search_text'].tolist()
        documents.append(user_input)
        
        tfidf_matrix = tfidf.fit_transform(documents)
        similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
        best_match_idx = similarities.argsort()[0][-1]
        best_score = similarities[0][best_match_idx]
        
        if best_score > 0.05:
             return univ_df.iloc[best_match_idx], "sim"
    except:
        pass
    return None, None

# ==========================================
# 3. 사용자 입력 받기 (UI)
# ==========================================
col1, col2 = st.columns(2)

with col1:
    grade_input = st.selectbox("진급할 학년을 선택하세요", [2, 3])

with col2:
    user_interest = st.text_input("관심 계열/학과/키워드 입력", placeholder="예: 기계, 의예, 컴공, 로봇")

search_btn = st.button("🔍 과목 추천받기", type="primary")

# ==========================================
# 4. 결과 출력 화면
# ==========================================
if search_btn and user_interest:
    best_major, match_type = find_best_major_smart(user_interest, univ_df)
    
    if best_major is None:
        st.error(f"😥 '{user_interest}'와 관련된 학과를 찾지 못했습니다.")
        st.info("조금 더 일반적인 단어나 정확한 학과명으로 다시 검색해보세요.")
    else:
        # 학과 찾음 성공 메시지
        st.success(f"🎉 **[{best_major['학과명']}]** 관련 정보를 찾았습니다!")
        
        # 대학 요구사항 파싱
        req_subjects = [x.strip() for x in str(best_major.get('필수이수과목(수학)','')).split(',') if x.strip() not in ['-', 'nan', '']] + \
                       [x.strip() for x in str(best_major.get('필수이수과목(과학)','')).split(',') if x.strip() not in ['-', 'nan', '']]
        
        rec_subjects = [x.strip() for x in str(best_major.get('권장이수과목(수학)','')).split(',') if x.strip() not in ['-', 'nan', '']] + \
                       [x.strip() for x in str(best_major.get('권장이수과목(과학/정보)','')).split(',') if x.strip() not in ['-', 'nan', '']]
        
        # 키워드 추출
        keywords = [x.strip() for x in str(best_major.get('관련키워드','')).split(',') if x.strip()]
        keywords.append(best_major['학과명'].replace("학과", "").replace("공학", "").replace("부", ""))

        # 대학 요구 정보 표시
        with st.expander("📌 대학에서 요구하는 과목 보기 (클릭)", expanded=True):
            st.markdown(f"**필수(⭐⭐⭐):** {', '.join(req_subjects) if req_subjects else '없음'}")
            st.markdown(f"**권장(⭐⭐):** {', '.join(rec_subjects) if rec_subjects else '없음'}")

        st.divider()
        st.subheader(f"🏫 {grade_input}학년 추천 과목 리스트")

        # 학교 데이터 필터링
        my_grade_subjects = school_df[school_df['학년'] == grade_input].copy()
        
        if my_grade_subjects.empty:
            st.warning("해당 학년의 데이터가 없습니다.")
        else:
            grouped = my_grade_subjects.groupby('선택군ID')
            
            for group_id, group_df in grouped:
                group_info = group_df.iloc[0]
                category = group_info['교과군']
                semester = group_info['학기']
                select_rule = group_info['비고(선택수)']
                
                # 카드 형태로 보여주기
                with st.container():
                    st.markdown(f"#### 📅 {semester}학기 | {category} ({select_rule})")
                    
                    result_rows = []
                    for _, subject in group_df.iterrows():
                        sub_name = subject['과목명']
                        sub_norm = normalize(sub_name)
                        
                        icon = ""
                        note = ""
                        highlight = False
                        
                        # 매칭 로직
                        is_match = False
                        
                        # 1. 필수
                        for req in req_subjects:
                            if normalize(req) in sub_norm:
                                icon = "⭐⭐⭐"
                                note = "필수 추천"
                                highlight = True
                                is_match = True
                                break
                        
                        # 2. 권장
                        if not is_match:
                            for rec in rec_subjects:
                                if normalize(rec) in sub_norm:
                                    icon = "⭐⭐"
                                    note = "권장 추천"
                                    highlight = True
                                    is_match = True
                                    break
                        
                        # 3. 키워드 AI 추천
                        if not is_match:
                            for key in keywords:
                                if len(key) >= 2 and key in sub_name:
                                    icon = "⭐"
                                    note = "AI 추천 (관련도 높음)"
                                    highlight = True
                                    is_match = True
                                    break
                        
                        if highlight:
                            result_rows.append(f"**{icon} {sub_name} ({note})**")
                        else:
                            result_rows.append(f"<span style='color:gray'>{sub_name}</span>")
                    
                    # 결과 출력 (HTML 태그 허용)
                    for row in result_rows:
                        st.markdown(f"- {row}", unsafe_allow_html=True)
                    
                    st.markdown("---")
