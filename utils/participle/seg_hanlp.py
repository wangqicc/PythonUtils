import hanlp
# Need to download
HanLP = hanlp.load(hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_BASE_ZH)


def hanlp_seg_str(data, tasks="pos/pku"):
    """
    HanLP Segment
    :param data: input text
    :param tasks: seg model
    :return: seg str
    """
    json_seg = HanLP(data=data, tasks=tasks)

    words = json_seg["tok/fine"]
    natures = json_seg[tasks]

    # concat
    seg_list = []
    for word, nature in zip(words, natures):
        seg_list.append(word+"/"+nature)

    seg_str = "\t".join(seg_list)
    return seg_str


if __name__ == '__main__':
    text = "下午好，@老杨头207959905  你关注的 @海峡新干线 开播啦：滚动直播：民进党上台五周年，台湾疫情大暴发，台北进入半封城>>"
    print(hanlp_seg_str(text))
