import requests  # http lib
from bs4 import BeautifulSoup  # climb lib
import os # operation system
import traceback # trace deviance

def download(url,filename):
    if os.path.exists(filename):
        print('file exists!')
        return
    try:
        r = requests.get(url,stream=True,timeout=60)
        r.raise_for_status()  #HTTPError异常
        with open(filename,'wb') as f:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk: # filter out keep-alove new chunks
                    f.write(chunk)
                    f.flush()
        return filename
    except KeyboardInterrupt: #用户中断执行操作
        if os.path.exists(filename):
            os.remove(filename) #移除路径
        return KeyboardInterrupt
    except Exception:
        traceback.print_exc() #捕获错误出现的文件路径
        if os.path.exists(filename):
            os.remove(filename)

if os.path.exists('imgs') is False:
    os.makedirs('imgs')

start = 1
end = 8000
for i in range(start, end+1):
    url = 'http://konachan.net/post?page=%d&tags=' % i
    html = requests.get(url).text # gain the web's information
    soup =  BeautifulSoup(html,'html.parser') # html的解析工具
    for img in soup.find_all('img',class_="preview"):# 遍历所有preview类，找到img标签
        target_url = img['src']
        filename = os.path.join('imgs',target_url.split('/')[-1])#返回用分隔符组合的新字符串
        download(target_url,filename)
    print('%d / %d' % (i,end))