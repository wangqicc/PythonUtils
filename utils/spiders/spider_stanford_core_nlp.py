import json
import time
import utils.util as util
import random
from requests_html import HTMLSession
session = HTMLSession()


def stanford_core_nlp_seg_str(data):
    """
    Stanford Core NLP Segment
    :param data: input text
    :return: seg str
    """
    # set random time
    time.sleep(random.random()*2)
    url = "https://corenlp.run/"
    params = {
        "properties": {
            "annotators": "tokenize,ssplit,pos,ner,regexner",
            "date": util.format_timestamp(timestamp=time.time())
        },
        "pipelineLanguage": "zh"
    }
    r = session.post(url=url, params=params, data=data.encode("utf-8"))

    seg_str = ""
    if r.status_code != 200:
        print("connect error!")
        return seg_str

    words = []
    natures = []
    list_seg = json.loads(r.text)["sentences"]
    for idx in range(len(list_seg)):
        seg = list_seg[idx]["tokens"]
        for item in seg:
            words.append(item["word"])
            natures.append(item["pos"])

    # concat
    seg_list = []
    for word, nature in zip(words, natures):
        seg_list.append(word+"/"+nature)

    seg_str = "\t".join(seg_list)
    return seg_str


if __name__ == '__main__':
    text = "下午好，@老杨头207959905  你关注的 @海峡新干线 开播啦：滚动直播：民进党上台五周年，台湾疫情大暴发，台北进入半封城>>"
    print(stanford_core_nlp_seg_str(text))
