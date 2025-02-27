import torch
import pandas as pd
import re
from transformers import T5Tokenizer, T5ForConditionalGeneration
from tqdm import tqdm
import mojimoji
from io import StringIO
import streamlit as st
import streamlit_ext as ste


def set_streamlit():
    # カスタムテーマの定義
    st.set_page_config(
        page_title="リスくん - リスク因子構造化システム",
        page_icon=":chipmunk:",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            "Get Help": "https://www.extremelycoolapp.com/help",
            "Report a bug": "https://www.extremelycoolapp.com/bug",
            "About": """
            # リスくん:chipmunk: リスク因子構造化システム
            脳卒中のリスク因子を構造化し、csv形式で出力するシステムです。

            GitHub: https://github.com/sociocom/risk-extractor
            """,
        },
    )
    st.title("リスくん:chipmunk: リスク因子構造化システム")
    st.markdown("###### 脳卒中のリスク因子を構造化し、csv形式で出力するシステムです。")

    st.sidebar.write(
        "### サンプルファイルで実行する場合は以下のファイルをダウンロードしてください"
    )
    sample_csv = pd.read_csv("data/sample3_utf8.csv")
    sample_csv = sample_csv.to_csv(index=False)
    ste.sidebar.download_button("sample data", sample_csv, f"riskun_sample.csv")

    st.sidebar.markdown("### 因子構造化に用いるcsvファイルを選択してください")
    # ファイルアップロード
    uploaded_file = st.sidebar.file_uploader(
        "Choose a CSV file", accept_multiple_files=False
    )
    return uploaded_file


def convert_to_utf8(content, encoding):
    try:
        return content.decode(encoding).encode("utf-8")
    except UnicodeDecodeError:
        return None


def read_uploaded_file_as_utf8(uploaded_file):
    # ファイルをバイナリモードで読み込み
    content = uploaded_file.read()

    # エンコーディングを自動検出し、UTF-8に変換
    encodings_to_try = [
        "utf-8",
        "shift-jis",
        "cp932",
        "latin-1",
        "ISO-8859-1",
        "euc-jp",
        "euc-kr",
        "big5",
        "utf-16",
    ]
    utf8_content = None

    for encoding in encodings_to_try:
        utf8_content = convert_to_utf8(content, encoding)
        if utf8_content is not None:
            break

    try:
        df = pd.read_csv(StringIO(utf8_content.decode("utf-8")))
    except pd.errors.EmptyDataError:
        st.error(
            "データが読み込めませんでした。utf-8のエンコードのcsvファイルを選んでください。"
        )

    return df


def replace_spaces(text):
    # 2つ以上連続したスペースを1つのスペースに置換
    text = re.sub(r" {2,}", " ", text)
    # タブをスペースに置換
    text = re.sub(r"\t", " ", text)
    return text


class GenerateText:
    """GenerateText"""

    def __init__(
        self,
        data_batch_size=16,
        token_max_length_src=512,
        token_max_length_tgt=8,
    ):
        self.data_batch_size = data_batch_size
        self.token_max_length_src = token_max_length_src
        self.token_max_length_tgt = token_max_length_tgt
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def download_model(self):
        """Download model"""
        tokenizer = T5Tokenizer.from_pretrained(f"models/tokenizer")
        model = T5ForConditionalGeneration.from_pretrained(f"models/model").to(
            self.device
        )
        return tokenizer, model

    def generate_text(self, model, tokenizer, df0, text, target_columns):
        """Generate text"""
        df = df0.copy()
        original_texts = df[text].to_list()
        prediction = []
        df[text] = df[text].apply(replace_spaces)
        df[text] = df[text].apply(mojimoji.han_to_zen)
        df[text] = str(target_columns) + "：" + df[text]

        for i in tqdm(range(0, len(df), self.data_batch_size)):
            batch = df.iloc[i : i + self.data_batch_size, :]
            soap = batch[text].to_list()
            generated_text = generate_text_from_model(
                tags=soap,
                trained_model=model,
                tokenizer=tokenizer,
                num_return_sequences=1,
                max_length_src=self.token_max_length_src,
                max_length_target=self.token_max_length_tgt,
                num_beams=10,
                device=self.device,
            )
            # original_texts.extend(soap)
            prediction.extend(generated_text)

        column_df = pd.DataFrame(
            {
                text: original_texts,
                target_columns: prediction,
            }
        )
        return column_df

def generate_text_from_model(
    tags,
    trained_model,
    tokenizer,
    num_return_sequences=1,
    max_length_src=30,
    max_length_target=300,
    num_beams=10,
    device="cpu",
):
    trained_model.eval()

    batch = tokenizer(
        tags,
        max_length=max_length_src,
        truncation=True,
        padding="longest",
        return_tensors="pt",
    )

    # 生成処理を行う
    # outputs = trained_model.model.generate(
    outputs = trained_model.generate(
        input_ids=batch["input_ids"].to(device),
        attention_mask=batch["attention_mask"].to(device),
        max_length=max_length_target,
        repetition_penalty=8.0,  # 同じ文の繰り返し（モード崩壊）へのペナルティ
        # temperature=1.0,  # 生成にランダム性を入れる温度パラメータ
        num_beams=num_beams,  # ビームサーチの探索幅
        # diversity_penalty=1.0,  # 生成結果の多様性を生み出すためのペナルティパラメータ
        # num_beam_groups=10,  # ビームサーチのグループ
        num_return_sequences=num_return_sequences,  # 生成する文の数
    )

    generated_texts = [
        tokenizer.decode(
            ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        for ids in outputs
    ]

    return generated_texts
