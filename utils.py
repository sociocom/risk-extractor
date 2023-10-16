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
