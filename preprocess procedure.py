import csv
import nltk
import re
import time
from nltk.corpus import stopwords
from flashtext import KeywordProcessor
from collections import Counter
def load_texts(filename1,filename2):
    with open (filename2, encoding= 'utf-8-sig') as f2:
        with open(filename1, encoding='utf-8-sig') as f1:
            specific_terms = f1.read().lower().splitlines() # reading specific terms of lexicon
            keywordprocessor = KeywordProcessor()
            terms_dict = keywordprocessor.add_keywords_from_list(specific_terms) #words tablet
            sentence = f2.read().lower()
            """'sentence_remove_url = re.sub(r'^(https:\S+)',' ', sentence) # removing URL
            remove_chars = '[0-9!"#$&%=\'()/\*+,.;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
            sentence = re.sub(remove_chars, '', sentence_remove_url) # removing punctuation"""
            CVD_content = sentence.splitlines() ## list
            noun_words_list =[]
            fin_list = []
            for i in range(len(CVD_content)):
                    starttiming = time.time()
                    specific_terms1 = keywordprocessor.extract_keywords(CVD_content[i])
                    for item in specific_terms1:
                        CVD_content[i] = CVD_content[i].replace(item, '')
                    uni_keyword = nltk.word_tokenize(CVD_content[i])  ## 分词 unigram tokenisation
                    uni_keyword_remove_url = re.sub(r'^(https:\S+)', ' ', str(uni_keyword))  # removing URL
                    remove_chars = '[0-9!"#$&%=\'()/\*+,.;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
                    uni_keyword_remove_punc = re.sub(remove_chars, '', uni_keyword_remove_url)  # removing punctuation
                    list_stopwords = list(set(stopwords.words('english')))
                    filtered_words = [w for w in uni_keyword_remove_punc if w not in list_stopwords]
                    noun_words = nltk.pos_tag(filtered_words)  # 标注词性
                    for noun_word in noun_words:
                        if 'NNP' == noun_word[1] or 'NN' == noun_word[1] or 'NNS' == noun_word[1]:  # noun_word 是元组，可以加入列表
                            noun_words_list.append(noun_word[0])
                    final_list = specific_terms1 + noun_words_list
                    wnl = nltk.WordNetLemmatizer()  ## 词形还原
                    final_list1 = [wnl.lemmatize(t) for t in final_list]  ### 词性归并，主要是删除词缀产生的词
                    final_list2 = list(filter(lambda s: isinstance(s, str) and len(s) > 2, final_list1))
                    fin_list.append(final_list2)
                    noun_words_list = []
                    #print(final_list2)
                    endtime = time.time()
                    print('processing the %d document took time：\t' % i, (endtime - starttiming))
            return fin_list
def top_common_text(list1,n):
    with open('uni_terms.csv', 'w', encoding= 'utf-8-sig') as f_terms:
        Fre_list = []
        com_words_list = []
        for i in range(len(list1)):
            for j in range(len(list1[i])):
                Fre_list.append(list1[i][j])
        Frequency = Counter(Fre_list)
        print( Frequency)
        uni_terms = Frequency.items()
        for item in uni_terms:
            csv_writer = csv.writer(f_terms)
            csv_writer.writerow([item[0],item[1]])
        com_words = Frequency.most_common(n)
        selected_specific = ['cell', 'hypertension', 'trial', 'heart failure','blood pressure', 'surgery', 'myocardial infarction','cohort', 'tissue', 'gene', 'atrial fibrillation']
        for i in range(len(com_words)):
            if com_words[i][0] not in selected_specific:
              com_words_list.append(com_words[i][0])

        return com_words_list
def clean_text(list1, list2):
    cleanfile = []
    cleanfile2 = []
    for i in range(len(list1)):
        for j in list1[i]:
            if j not in list2:
                cleanfile.append(j)
        cleanfile1 = cleanfile
        cleanfile = []
        cleanfile2.append(cleanfile1)
    return cleanfile2 ## there is a problem.
if __name__ == '__main__':
    starttiming = time.time()
    load_list = load_texts('C:/Users/qianqianxieqqx/PycharmProjects/pythonProject/CVD terms.txt', 'C:/Users/qianqianxieqqx/PycharmProjects/pythonProject/10 years data 142 clusters ABTI.txt')
    common_list = top_common_text(load_list, 1)
    fianl_clean_text = clean_text(load_list, common_list)
    with open('clean_test_data_solidli.txt', 'w', encoding='UTF-8-sig', newline='')as f:
        for i in range(len(fianl_clean_text)):
            for j in range(len(fianl_clean_text[i])):
                f.write(fianl_clean_text[i][j] + ' ')
            f.write(' \n')
    endtime = time.time()
    print(endtime-starttiming)