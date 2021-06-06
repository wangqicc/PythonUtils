import re
from datetime import datetime


def read_lines(filename):
    """
    read file to list
    :param filename: file name
    :return: list
    """
    lines = []
    with open(filename, "r", encoding="UTF-8") as f:
        for line in f.readlines():
            line = re.sub(r"\n$", "", line, 0, re.I | re.M)
            lines.append(line)
    return lines


def format_timestamp(timestamp, fmt="%Y-%m-%dT%H:%M:%S"):
    """
    Format timestamp
    :param timestamp: input timestamp
    :param fmt: formatting style
    :return: format str
    """
    return datetime.fromtimestamp(int(timestamp)).strftime(fmt)
