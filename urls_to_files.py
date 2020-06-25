from datetime import datetime
import time, cv2, itertools, os, json
import numpy as np
import pandas as pd
from PIL import Image

from urllib.request import urlopen

def url_to_image(url):
    req = urlopen(url)
    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img
    ''' End Function '''

def main():
    file_path = ''
    dataset = pd.read_csv(file_path, header=True)
    


if __name__ == "__main__":
    main()