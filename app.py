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
# 1. 데이터 로드
# ==========================================
@st.cache_data
def load_data():
    try:
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
# 2. 로직 함수들
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
# 3. 사용자 입력
# ==========================================
col1, col2 = st.columns(2)
with col1:
    grade_input = st.selectbox("진급할 학년을 선택하세요", [2, 3])
with col2:
    user_interest = st.text_input("관심 계열/학과/키워드 입력", placeholder="예: 경영, 기계, 의예, 컴공")

search_btn = st.button("🔍 과목 추천받기", type="primary")

# ==========================================
# 4. 결과 출력 로직
# ==========================================
if search_btn and user_interest:
    best_major, match_type = find_best_major_smart(user_interest, univ_df)
    
    if best_major is None:
        st.error(f"😥 '{user_interest}'와 관련된 학과를 찾지 못했습니다.")
        st.info("조금 더 일반적인 단어나 정확한 학과명으로 다시 검색해보세요.")
    else:
        st.success(f"🎉 **[{best_major['학과명']}]** 관련 정보를 찾았습니다!")
        
        # 대학 요구사항 파싱
        req_subjects = [x.strip() for x in str(best_major.get('필수이수과목(수학)','')).split(',') if x.strip() not in ['-', 'nan', '']] + \
                       [x.strip() for x in str(best_major.get('필수이수과목(과학)','')).split(',') if x.strip() not in ['-', 'nan', '']]
        
        rec_subjects = [x.strip() for x in str(best_major.get('권장이수과목(수학)','')).split(',') if x.strip() not in ['-', 'nan', '']] + \
                       [x.strip() for x in str(best_major.get('권장이수과목(과학/정보)','')).split(',') if x.strip() not in ['-', 'nan', '']]
        
        # 키워드
        keywords = [x.strip() for x in str(best_major.get('관련키워드','')).split(',') if x.strip()]
        keywords.append(best_major['학과명'].replace("학과", "").replace("공학", "").replace("부", ""))

        with st.expander("📌 대학에서 요구하는 과목 보기 (클릭)", expanded=False):
            st.markdown(f"**필수(⭐⭐⭐):** {', '.join(req_subjects) if req_subjects else '없음'}")
            st.markdown(f"**권장(⭐⭐):** {', '.join(rec_subjects) if rec_subjects else '없음'}")

        st.divider()
        st.subheader(f"🏫 {grade_input}학년 추천 과목 리스트")

        # ------------------------------------------------
        # [기능 1] 현재 학년 추천 로직
        # ------------------------------------------------
        my_grade_subjects = school_df[school_df['학년'] == grade_input].copy()
        total_match_count = 0  # 전체 추천 개수 카운트

        if my_grade_subjects.empty:
            st.warning("해당 학년의 데이터가 없습니다.")
        else:
            grouped = my_grade_subjects.groupby('선택군ID')
            
            for group_id, group_df in grouped:
                group_info = group_df.iloc[0]
                category = group_info['교과군']
                semester = group_info['학기']
                select_rule = group_info['비고(선택수)']
                
                # 그룹별 결과를 담을 리스트
                group_results = []
                
                for _, subject in group_df.iterrows():
                    sub_name = subject['과목명']
                    sub_norm = normalize(sub_name)
                    
                    icon = ""
                    note = ""
                    highlight = False
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
                    
                    # 3. AI 키워드
                    if not is_match:
                        for key in keywords:
                            if len(key) >= 2 and key in sub_name:
                                icon = "⭐"
                                note = "AI 추천"
                                highlight = True
                                is_match = True
                                break
                    
                    if highlight:
                        total_match_count += 1
                        group_results.append(f"**{icon} {sub_name} ({note})**")
                    else:
                        group_results.append(f"<span style='color:gray'>{sub_name}</span>")
                
                # 화면 출력 (카드 형태)
                with st.container():
                    st.markdown(f"#### 📅 {semester}학기 | {category} ({select_rule})")
                    for row in group_results:
                        st.markdown(f"- {row}", unsafe_allow_html=True)
                    st.markdown("---")

        # ------------------------------------------------
        # [기능 2] 추천 과목이 0개일 때 안내 문구
        # ------------------------------------------------
        if total_match_count == 0:
            st.info("💡 **현재 학년에서는 필수 이수과목이나 권장 이수과목이 없습니다.** \n\n부담 갖지 말고 듣고 싶은 과목을 자유롭게 선택해보세요! 😊")

        # ------------------------------------------------
        # [기능 3] 2학년일 경우, 3학년 과목 미리보기 (Only 추천 과목만)
        # ------------------------------------------------
        if grade_input == 2:
            st.write("") # 여백
            st.subheader("👀 (미리보기) 3학년 때는 이런 과목이 있어요!")
            
            next_grade_subjects = school_df[school_df['학년'] == 3].copy()
            next_match_list = []

            for _, subject in next_grade_subjects.iterrows():
                sub_name = subject['과목명']
                sub_norm = normalize(sub_name)
                match_grade = ""
                match_note = ""
                
                is_match = False
                
                # 필수 체크
                for req in req_subjects:
                    if normalize(req) in sub_norm:
                        match_grade = "⭐⭐⭐"
                        match_note = "필수"
                        is_match = True
                        break
                
                # 권장 체크
                if not is_match:
                    for rec in rec_subjects:
                        if normalize(rec) in sub_norm:
                            match_grade = "⭐⭐"
                            match_note = "권장"
                            is_match = True
                            break

                # 키워드 체크
                if not is_match:
                     for key in keywords:
                        if len(key) >= 2 and key in sub_name:
                            match_grade = "⭐"
                            match_note = "관련"
                            is_match = True
                            break
                
                if is_match:
                    next_match_list.append(f"{match_grade} **{sub_name}** ({match_note})")

            if next_match_list:
                with st.expander("📚 3학년 주요 추천 과목 열어보기"):
                    for item in next_match_list:
                        st.markdown(f"- {item}")
            else:
                st.write("3학년 과목 중에서도 특별히 매칭되는 과목은 없네요! 폭넓게 탐색해보세요.")
