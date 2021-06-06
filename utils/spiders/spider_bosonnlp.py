import json
import time
import random
from requests_html import HTMLSession
session = HTMLSession()


def bosonnlp_seg_str(data):
    """
    BosonNLP Segment
    :param data: input text
    :return: seg str
    """
    # set random time
    time.sleep(random.random()*2)
    url = "http://static.bosonnlp.com/analysis/tag/"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/90.0.4430.212 Safari/537.36",
        "Referer": "http://static.bosonnlp.com/demo",
        "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8"
    }
    params = {}
    data = "data=" + data
    data = data.encode("utf-8")
    r = session.post(url=url, params=params, data=data, headers=headers, allow_redirects=True)

    seg_str = ""
    if r.status_code != 200:
        print("connect error!")
        return seg_str

    json_seg = json.loads(r.text)[0]
    words = json_seg["word"]
    natures = json_seg["tag"]

    # concat
    seg_list = []
    for word, nature in zip(words, natures):
        seg_list.append(word+"/"+nature)

    seg_str = "\t".join(seg_list)
    return seg_str


if __name__ == '__main__':
    text = "下午好，@老杨头207959905  你关注的 @海峡新干线 开播啦：滚动直播：民进党上台五周年，台湾疫情大暴发，台北进入半封城>>"
    print(bosonnlp_seg_str(text))
