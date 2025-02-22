import os
import pandas as pd
import torch
from datetime import datetime
from utils import GenerateText, set_streamlit, read_uploaded_file_as_utf8
import streamlit as st
import streamlit_ext as ste


torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)]

# or simply:
torch.classes.__path__ = []

# def main():
uploaded_file = set_streamlit()

generator = GenerateText(
    data_batch_size=4,
    token_max_length_src=512,
    token_max_length_tgt=8,
)
target_columns = [
    "性別",
    "身長",
    "体重",
    "年齢",
    "HbA1c",
    "CRP",
    "血圧",
    "体温",
    "脈拍",
    "抗血小板薬",
    "抗凝固薬",
    "スタチン",
    "糖尿病治療薬",
    "糖尿病",
    "喫煙",
    "飲酒",
    "診断名",
]

if uploaded_file:
    df = read_uploaded_file_as_utf8(uploaded_file)
    st.write("入力ファイル (先頭5件までを表示)")
    st.dataframe(df.head(5))
    text = st.selectbox(
        "カルテが記載されている項目を選んでください",
        (df.columns),
        index=None,
        placeholder="Select...",
    )
    st.write("選択した項目:", text)

    if text:
        with st.spinner("実行中..."):
            tokenizer, model = generator.download_model()
            for i, column in enumerate(target_columns):
                column_df = generator.generate_text(model, tokenizer, df, text, column)
                display_df = column_df.copy()
                display_df[text] = display_df[text].iloc[:5].str[:5] + "..."

                if i == 0:
                    output_df = column_df
                    mytable = st.table(display_df.iloc[:5].T)
                else:
                    output_df = pd.merge(output_df, column_df, on=text, how="inner")
                    mytable.add_rows(display_df[[column]].iloc[:5].T)

        st.write("出力結果 (先頭5件までを表示)")
        st.dataframe(output_df.head(5))
        if "completed" not in st.session_state:
            st.session_state["completed"] = True

        file_name = uploaded_file.name.replace(".csv", "").replace("riskun_", "")
        timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        csv = output_df.to_csv(index=False)
        # b64 = base64.b64encode(csv.encode("utf-8-sig")).decode()
        # href = f'<a href="data:application/octet-stream;base64,{b64}" download="{file_name}-riskun-{timestamp}.csv">Download Link</a>'
        # st.markdown(f"CSVファイルのダウンロード: {href}", unsafe_allow_html=True)

        ste.download_button(
            "Click to download data", csv, f"{file_name}-riskun-{timestamp}.csv"
        )

# if __name__ == "__main__":
# main()
