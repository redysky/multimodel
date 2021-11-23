#文本预处理的程序---去停用词,处理后转存到一个文件夹中--text
from collections import Counter
import jieba
import os
import chardet


def text_process():
	cur_path = os.path.dirname(__file__)  # 获取当前文件路径
	parent_path = os.path.dirname(cur_path)  # 获取当前文件夹父目录
	stop_list = "./data/去停用词.txt"
	# 需要自己制作一个所有文本(txt)的集合，就是放在一个文件夹(o_path)中
	o_path = os.path.join(parent_path, r'data//Unprocessed_text/')
	# 最终的文本数据路径
	f_path=os.path.join(parent_path, 'data/text/')
	outstr=''
	stopwords=[' ','(',')','」','「',"'",'^','|']
	for line in open(stop_list,encoding='utf-8',errors='ignore').readlines():
		stopwords.append(line.strip())

	list_text=[]
	for i in os.listdir(o_path):
		list_text.append(i)
	# print(list_text)

	for i in range(len(list_text)):
		if chardet.detect(open(os.path.join(o_path+list_text[i]),'rb').read())['encoding']=='GB2312':
			for line in open(os.path.join(o_path+list_text[i]),encoding='ansi',errors='ignore'):
				for word in line:
					if word not in stopwords:
						if word !='\t':
							outstr += word
				open(os.path.join(f_path+list_text[i]),'w',encoding='utf-8-sig').write(outstr+'\n')
			outstr=''
		else:
			for line in open(os.path.join(o_path+list_text[i]),encoding='utf-8-sig',errors='ignore'):
				for word in line:
					if word not in stopwords:
						if word !='\t':
							outstr += word
				open(os.path.join(f_path+list_text[i]),'w',encoding='utf-8-sig').write(outstr+'\n')
			outstr=''


text_process()