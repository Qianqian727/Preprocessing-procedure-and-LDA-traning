import nltk
import time
import csv
import numpy as np
import pyLDAvis
from flashtext import KeywordProcessor
from gensim import corpora, models
import pyLDAvis.gensim_models
from gensim.models.coherencemodel import CoherenceModel
from gensim.test.utils import datapath
def load_texts(filename1,filename2):
    with open (filename1, encoding= 'utf-8-sig') as f1:
        with open(filename2, encoding='utf-8-sig') as f2:
            specific_terms = f2.read().lower().splitlines()
            keywordprocessor = KeywordProcessor()
            terms_dict = keywordprocessor.add_keywords_from_list(specific_terms) #words tablet
            sentence = f1.read()
            CVD_content = sentence.splitlines()  ## list
            fin_list = []
            for i in range(len(CVD_content)):
                starttiming = time.time()
                specific_terms1 = keywordprocessor.extract_keywords(CVD_content[i])
                for item in specific_terms1:
                    CVD_content[i] = CVD_content[i].replace(item, ' ')
                uni_keyword = nltk.word_tokenize(CVD_content[i])  ## 分词 unigram list
                uni_keyword1= list(filter(lambda s: isinstance(s,str) and len(s) >2, uni_keyword))
                final_list = specific_terms1 + uni_keyword1
                fin_list.append(final_list)
                endtime = time.time()
                print('processing the %d document took time：\t' % i, (endtime - starttiming))
            return fin_list
def topic_model(clean_list,filename1, filename2):
        len_document = len(clean_list)
        dictionary1 = corpora.Dictionary(clean_list)  ##生成词典 将l生成了词典, 得到单词出现的次数以及统计信息
        dictionary1.filter_extremes(no_below= 10, no_above= 0.9, keep_n=100000)
        print(dictionary1)
        corpus = [dictionary1.doc2bow(text1) for text1 in clean_list]  # 每个text对应的稀疏向量 计算文本向量 corpus 中的l是列表形式，经过处理之后的列表

        lda = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary1, iterations=10000,
                              random_state= 1000,alpha=0.067, eta=0.1, minimum_probability=0.001, #
                              update_every=1, chunksize=5000, passes=10)  # LDA 模型

        temp_file = datapath("C:/Users/qianqianxieqqx/PycharmProjects/pythonProject/lda_model")
        lda.save(temp_file) ### save LDA model
        cm = CoherenceModel(model= lda, corpus=corpus, coherence='u_mass')
        coherence = cm.get_coherence()
        print(coherence)
        doc_topic = [a for a in lda[corpus]]  # 所有文档的主题
        doc_topics = lda.get_document_topics(corpus, minimum_probability= 0.00, minimum_phi_value= 0.00, per_word_topics=False)  # 所有文档的主题分布
        #doc_topics = lda.get_document_topics(corpus)
        idx = np.arange(len_document)
        with open(filename1, 'w', encoding='UTF-8-sig', newline='')as f1:
            csv_writer = csv.writer(f1)
            csv_writer.writerow(
                ["", "topic 0", "topic1", "topic2", "topic3", "topic4", "topic5", "topic6", "topic7", "topic8",
                 "topic9"])
            for i in idx:
                topic = np.array(doc_topics[i])
                topic_distribute = np.array(topic[:, 1])  ## 主题—文档概率的分布
                topic_distribute = list(topic_distribute)
                topic_distribute.insert(0, 'doc_%d:' % i)
                np.array(topic_distribute)
                csv_writer.writerow(topic_distribute)
        with open(filename2, 'w', encoding='UTF-8-sig', newline='') as f2:
            csv_writer = csv.writer(f2)
            list1 = []
            for topic_id in range(num_topics):
                term_distribute_all = lda.get_topic_terms(topicid=topic_id,topn=40)  # 所有词的词分布, list , 二元组(260,0.00258379)
                term_distribute = term_distribute_all[:num_show_term]  # 只显示前几个词
                term_distribute = np.array(term_distribute)  # 转换成数组
                term_id = term_distribute[:, 0].astype(np.int64)  # 行取全部，列取0,term_id 是每个term对应的地址，类似指针， 260
                for t in term_id:
                    list1.append(dictionary1.id2token[t])
                list2 = list(term_distribute[:, 1])
                list3 = list(zip(list1, list2))
                list3.insert(0, ('Topic%d:' % topic_id, ''))
                for elem in list3:
                    csv_writer.writerow([elem[0], elem[1]])
                list1 = []
        '''with open(filename3, 'w', encoding='UTF-8-sig', newline='') as f3:
            csv_writer = csv.writer(f3)
            csv_writer.writerow(["keywords", "topic1", "topic2", "topic3", "topic4", "topic5"])
            list4 = []
            for item_id in range(num_items):
                term_distribute1 = lda.get_term_topics(item_id, minimum_probability=0.00000000)
                list1 = [dictionary1.id2token[item_id]]
                for i in range(len(term_distribute1)):
                    list4.append(term_distribute1[i][1])
                list3 = [list1 + list4]
                for elem in list3:
                    csv_writer.writerow(elem)
                   # csv_writer.writerow([elem[0],elem[1], elem[2], elem[3], elem[4], elem[5], elem[6], elem[7], elem[8], elem[9],elem[10], elem[11], elem[12], elem[13], elem[14], elem[15], elem[16], elem[17], elem[18], elem[19],elem[20]])
                list4 = []'''
        prepared = pyLDAvis.gensim_models.prepare(lda, corpus, dictionary1)
        # pyLDAvis.show(prepared, open_browser=True)
        pyLDAvis.save_html(prepared, 'lda_test_15_p1.html')
if __name__ == '__main__':
    starttiming = time.time()
    num_topics = 15 # 定义主题数
    num_show_topic = 5  # 每个文档显示前几个主题
    num_show_term = 40  # 每个主题显示几个词
    final_content = load_texts('test LDA_P1_10years.txt', 'keywords dict.txt')
    topic_model(final_content, 'topics-documents_distrubition_solid_li_15.csv', 'topic_words_solid_li_15.csv')
    endtime = time.time()
    print((endtime - starttiming))