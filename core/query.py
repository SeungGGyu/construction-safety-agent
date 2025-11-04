# 우리 : 공사종류(대분류) + 공종(중분류) + 작업프로세스 + 사고원인 조합
# A3 : 공사종류(대분류) + 공종(중분류) + 사고원인 조합

import pandas as pd

train = pd.read_csv('/home/user/Desktop/jiseok/capstone/RAG/construction-safety-agent/data/train_preprocessing.csv')
test = pd.read_csv('/home/user/Desktop/jiseok/capstone/RAG/construction-safety-agent/data/test_preprocessing.csv')

data_train = train.apply(
    lambda row: {
        "process": row["작업프로세스"],
        "construct_type": row["공종(중분류)"],
        "object_type": row["사고객체(중분류)"],
        "reason": row["사고원인"],
        "situation": f"'{row['공사종류(대분류)']}' 공사 중 '{row['공종(중분류)']}'의 '{row['작업프로세스']}' 과정에서 '{row['사고원인']}'으로 인해 사고가 발생하였습니다.",
    },
    axis=1
)

# DataFrame으로 변환
data_train = pd.DataFrame(list(data_train))

# Train 입력 데이터 설정
query = data_train["situation"].tolist()  # 질문 데이터 리스트로 변환

# pdf 뽑아낼 construct_type_query 설정
construct_type_query = data_train["construct_type"].tolist()  # "공종(중분류)"을 construct_type_query로 사용