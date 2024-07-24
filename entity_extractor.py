# -*- coding: utf-8 -*-
# @Author : maxin-nk
# @Email : xxin_ma@163.com
# @File : entity_extractor
from gensim.models import KeyedVectors
# import joblib
import os
import ahocorasick
import jieba
import numpy as np
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from transformers import BertTokenizer, BertModel
import torch
from sklearn.preprocessing import MinMaxScaler
import joblib


class EntityExtractor:
    def __init__(self):
        print('EntityExtractor.__init__()......')

        cur_dir = '/'.join(os.path.abspath(__file__).split('/')[:-1])
        # 路径
        self.vocab_path = os.path.join(cur_dir, 'data/vocab.txt')
        self.stopwords_path =os.path.join(cur_dir, 'data/stop_words.utf8')
        self.word2vec_path = os.path.join(cur_dir, 'data/merge_sgns_bigram_char300.txt')  # 预训练词向量
        # self.same_words_path = os.path.join(cur_dir, 'data/HIT-IRLab-同义词词林（扩展版）_full_2005.3.3.txt')
        self.stopwords = [w.strip() for w in open(self.stopwords_path, 'r', encoding='utf8') if w.strip()]

        # 意图分类模型文件
        # 如果报错No module named 'sklearn.externals.joblib'，则说明你使用了预加载的数据集（scikit_learn_data）是低版本sklearn下载的，当使用sklearn dataset抓取数据集时也会报这个错误，此时你可以将数据集删掉，重新下载即可。
        # self.tfidf_path = os.path.join(cur_dir, 'model/tfidf_model.m')
        self.nb_path = os.path.join(cur_dir, 'model/nb_classifier.joblib')  # 朴素贝叶斯模型
        # self.tfidf_model = joblib.load(self.tfidf_path)
        # self.nb_model = joblib.load(self.nb_path)
        self.tfidf_model = TfidfVectorizer()
        self.nb_model = joblib.load(self.nb_path)

        # self.disease_path = data_dir + 'disease_vocab.txt'
        # self.symptom_path = data_dir + 'symptom_vocab.txt'
        # self.alias_path = data_dir + 'alias_vocab.txt'
        # self.complication_path = data_dir + 'complications_vocab.txt'
        self.disease_path = 'data/' + 'disease_vocab.txt'
        self.symptom_path = 'data/' + 'symptom_vocab.txt'
        self.alias_path = 'data/' + 'alias_vocab.txt'
        self.complication_path = 'data/' + 'complications_vocab.txt'

        self.disease_entities = [w.strip() for w in open(self.disease_path, encoding='utf8') if w.strip()]
        self.symptom_entities = [w.strip() for w in open(self.symptom_path, encoding='utf8') if w.strip()]
        self.alias_entities = [w.strip() for w in open(self.alias_path, encoding='utf8') if w.strip()]
        self.complication_entities = [w.strip() for w in open(self.complication_path, encoding='utf8') if w.strip()]

        self.region_words = list(set(self.disease_entities+self.alias_entities+self.symptom_entities))

        # 构造领域actree
        self.disease_tree = self.build_actree(list(set(self.disease_entities)))
        self.alias_tree = self.build_actree(list(set(self.alias_entities)))
        self.symptom_tree = self.build_actree(list(set(self.symptom_entities)))
        self.complication_tree = self.build_actree(list(set(self.complication_entities)))

        # 用户提问的与实体相关的关键词(与意图相关的口语化的特征词)
        self.symptom_qwds = ['什么症状', '哪些症状', '症状有哪些', '症状是什么', '什么表征', '哪些表征', '表征是什么', '什么现象', '哪些现象', '现象有哪些', '症候', '什么表现', '哪些表现', '表现有哪些', '什么行为', '哪些行为', '行为有哪些', '什么状况', '哪些状况', '状况有哪些', '现象是什么', '表现是什么', '行为是什么']  # 询问症状
        self.cureway_qwds = ['药', '药品', '用药', '胶囊', '口服液', '炎片', '吃什么药', '用什么药', '怎么办', '买什么药', '怎么治疗', '如何医治', '怎么医治', '怎么治', '怎么医', '如何治', '医治方式', '疗法', '咋治', '咋办', '咋治', '治疗方法']  # 询问治疗方法
        self.lasttime_qwds = ['周期', '多久', '多长时间', '多少时间', '几天', '几年', '多少天', '多少小时', '几个小时', '多少年', '多久能好', '痊愈', '康复']  # 询问治疗周期
        self.cureprob_qwds = ['多大概率能治好', '多大几率能治好', '治好希望大么', '几率', '几成', '比例', '可能性', '能治', '可治', '可以治', '可以医', '能治好吗', '可以治好吗', '会好吗', '能好吗', '治愈吗']  # 询问治愈率
        self.check_qwds = ['检查什么', '检查项目', '哪些检查', '什么检查', '检查哪些', '项目', '检测什么', '哪些检测', '检测哪些', '化验什么', '哪些化验', '化验哪些', '哪些体检', '怎么查找', '如何查找', '怎么检查', '如何检查', '怎么检测', '如何检测']  # 询问检查项目
        self.belong_qwds = ['属于什么科', '什么科', '科室', '挂什么', '挂哪个', '哪个科', '哪些科']  # 询问科室
        self.disase_qwds = ['什么病', '啥病', '得了什么', '得了哪种', '怎么回事', '咋回事', '回事', '什么情况', '什么问题', '什么毛病', '啥毛病', '哪种病', '什么是']  # 询问疾病

    def build_actree(self, wordlist):
        """
        构造actree，加速过滤(算法是根据用户query，相当于遍历每个列表元素（词库），来判断每个元素是否存在于用户query中。)
        :param wordlist:实体词的列表(本项目仅处理了4种实体：disease/alias/symptom/complication)
        :return:
        """
        print('EntityExtractor.build_actree()......')

        # 建字典树(是一种很特别的树状信息检索数据结构：字符串多模匹配算法-AC自动机-Aho-Corasick算法)(https://www.cnblogs.com/vipsoft/p/17722761.html)
        # 1.高性能：相比传统的暴力搜索，Aho-Corasick算法极大提升了查找效率
        # 2.易用性：简洁明了的Python接口，便于集成到现有项目中
        # 3.稳定性：该库经过多次测试和优化，确保在大数据量下的可靠性和稳定性
        # 4.跨平台：全Python实现，可在任何支持Python的平台上运行
        actree = ahocorasick.Automaton()

        # 向树中添加单词
        for index, word in enumerate(wordlist):
            actree.add_word(word, (index, word))

        # 将actree转换为Aho_Corasick自动机以启用Aho_Corasick搜索(这不需要额外的内存)。
        actree.make_automaton()

        return actree

    def entity_reg(self, question):
        """
        query模式匹配, 得到匹配的词和类型。{'Disease':疾病名, 'Alias':疾病别名, 'Complication':并发症, 'Symptom':症状}
        :param question:用户的query (str)
        :return:
        """
        print('EntityExtractor.entity_reg()......')

        self.result = {}

        # 从之前构建的实体树中匹配出query种相应的实体元素
        # self.disease_tree.iter(question)==(query中词的结束下标,(建树列表中对应实体词的下标, '实体'))
        for i in self.disease_tree.iter(question):
            word = i[1][1]
            if "Disease" not in self.result and len(word) >= 2:  # 检查result字典中是否有Disease的key
                self.result["Disease"] = [word]
            elif len(word) >= 2:
                self.result["Disease"].append(word)

        for i in self.alias_tree.iter(question):
            word = i[1][1]
            if "Alias" not in self.result and len(word) >= 2:
                self.result["Alias"] = [word]
            elif len(word) >= 2:
                self.result["Alias"].append(word)

        for i in self.symptom_tree.iter(question):
            word = i[1][1]
            if "Symptom" not in self.result and len(word) >= 2:
                self.result["Symptom"] = [word]
            elif len(word) >= 2:
                self.result["Symptom"].append(word)

        for i in self.complication_tree.iter(question):
            word = i[1][1]
            if "Complication" not in self.result and len(word) >= 2:
                self.result["Complication"] = [word]
            elif len(word) >= 2:
                self.result["Complication"] .append(word)

        return self.result

    def find_sim_words(self, question):
        """
        当全匹配失败时，就采用相似度计算来找相似的词
        :param question:
        :return:
        """
        print('EntityExtractor.find_sim_words()......')

        # 加载自建的分词词典
        jieba.load_userdict(self.vocab_path)
        self.model = KeyedVectors.load_word2vec_format(self.word2vec_path, binary=False)

        # re.escape(): 可以对字符串中所有可能被解释为正则运算符的字符进行转义的应用函数
        # string.punctuation: !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
        sentence = re.sub("[{}]", re.escape(string.punctuation), question)
        sentence = re.sub("[，。‘’；：？、！【】]", " ", sentence)
        sentence = sentence.strip()

        # 要求划分所得的词：1.无空格，2.非停用词，3.长度≥2
        words = [w.strip() for w in jieba.cut(sentence) if w.strip() not in self.stopwords and len(w.strip()) >= 2]

        alist = []

        for word in words:
            temp = [self.disease_entities, self.alias_entities, self.symptom_entities, self.complication_entities]

            # 逐一计算word与Disease、Alias、Sympton和Complication列表中各单词的相似度
            for i in range(len(temp)):
                flag = ''
                if i == 0:
                    flag = "Disease"
                elif i == 1:
                    flag = "Alias"
                elif i == 2:
                    flag = "Symptom"
                else:
                    flag = "Complication"
                scores = self.simCal(word, temp[i], flag)
                alist.extend(scores)

            # TODO：每个词都要添加一下吗？
            # alist: [('实体词',相似度,'Disease'), ('实体词',相似度,'Alias'), ('实体词',相似度,'Symptom'), ('实体词',相似度,'Complication'), ...]
            # sorted(reverse=True):返回一个新列表，按降序包含可迭代对象中的所有项目。
            # sorted(alist, key=lambda k: k[1], reverse=True): alist元素按相似度大小降序排列
            temp1 = sorted(alist, key=lambda k: k[1], reverse=True)
            if temp1:
                self.result[temp1[0][2]] = [temp1[0][0]]

    def editDistanceDP(self, s1, s2):
        """
        采用DP方法计算编辑距离:对两个文本s1和s2，将s1经过以下操作(插入一个字符,删除一个字符,替换一个字符)得到s2，所使用的最少操作数
        :param s1:
        :param s2:
        :return:
        """
        print('EntityExtractor.editDistanceDP()......')

        m, n = len(s1), len(s2)

        if m*n == 0:
            return 0

        solution = [[0 for j in range(n + 1)] for i in range(m + 1)]
        for i in range(len(s2) + 1):
            solution[0][i] = i
        for i in range(len(s1) + 1):
            solution[i][0] = i

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i - 1] == s2[j - 1]:
                    solution[i][j] = solution[i - 1][j - 1]
                else:
                    solution[i][j] = 1 + min(solution[i][j - 1], min(solution[i - 1][j], solution[i - 1][j - 1]))
        return solution[m][n]

    def simCal(self, word, entities, flag):
        """
        计算词语和字典中的词的相似度：三种相似度结果取均值((相同字符的个数/min(|A|,|B|)), (cos), (1-编辑距离))
        :param word: query中划分得到的词(str)
        :param entities:Disease、Alias、Symptom或Complication实体类型对应的实体词库
        :return:(entity, score, flag)
        """
        print('EntityExtractor.simCal()......')

        a = len(word)
        scores = []
        for entity in entities:
            sim_num = 0
            b = len(entity)
            c = len(set(entity+word))
            temp = []

            # 统计词中相同的字母(字符)数，计算：相同字符的个数/min(|A|,|B|)
            for w in word:
                if w in entity:
                    sim_num += 1
            if sim_num != 0:
                score1 = sim_num / c  # overlap score
                temp.append(score1)
            try:
                score2 = self.model.similarity(word, entity)  # 余弦相似度分数
                temp.append(score2)
            except:
                pass
            score3 = 1 - self.editDistanceDP(word, entity) / (a + b)  # 编辑距离分数
            if score3:
                temp.append(score3)

            score = sum(temp) / len(temp)
            if score >= 0.7:
                scores.append((entity, score, flag))

        scores.sort(key=lambda k: k[1], reverse=True)
        return scores

    def check_words(self, wds, sent):
        """
        基于特征词分类意图
        :param wds:
        :param sent:
        :return:
        """
        print('EntityExtractor.check_words()......')

        for wd in wds:
            if wd in sent:
                return True
        return False

    def tfidf_features(self, text, vectorizer):
        """
        提取问题的TF-IDF特征
        :param text:用户的query文本
        :param vectorizer:TfidfVectorizer()
        :return:
        """
        print('EntityExtractor.tfidf_features()......')

        jieba.load_userdict(self.vocab_path)
        words = [w.strip() for w in jieba.cut(text) if w.strip() and w.strip() not in self.stopwords]
        sents = [' '.join(words)]

        if sents == ['']:
            return sents

        # tfidf = vectorizer.fit_transform(sents).toarray()
        vectorizer.fit_transform(sents)

        return vectorizer.get_feature_names_out()

    def other_features(self, text):
        """
        提取问题的关键词特征
        :param text:
        :return:
        """
        print('EntityExtractor.other_features()......')

        features = [0] * 7

        # 询问疾病
        for d in self.disase_qwds:
            if d in text:
                features[0] += 1

        # 询问症状
        for s in self.symptom_qwds:
            if s in text:
                features[1] += 1

        # 询问治疗方法
        for c in self.cureway_qwds:
            if c in text:
                features[2] += 1

        # 询问检查项目
        for c in self.check_qwds:
            if c in text:
                features[3] += 1

        # 询问治疗周期
        for p in self.lasttime_qwds:
            if p in text:
                features[4] += 1

        # 询问治愈率
        for r in self.cureprob_qwds:
            if r in text:
                features[5] += 1

        # 询问科室
        for d in self.belong_qwds:
            if d in text:
                features[6] += 1

        m = max(features)
        n = min(features)
        normed_features = []
        if m == n:
            normed_features = features
        else:
            for i in features:
                j = (i - n) / (m - n)
                normed_features.append(j)

        return np.array(normed_features)

    def encode_texts(self, texts, tokenizer, max_length=128):
        input_ids = []
        attention_masks = []

        for text in texts:
            encoded_dict = tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=max_length,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt',
            )

            input_ids.append(encoded_dict['input_ids'])
            attention_masks.append(encoded_dict['attention_mask'])

        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)

        return input_ids, attention_masks

    def get_bert_embeddings(self, inputs, masks, model):
        with torch.no_grad():
            outputs = model(inputs, attention_mask=masks)
            hidden_states = outputs.last_hidden_state
            # 使用句子的[CLS] token表示
            embeddings = hidden_states[:, 0, :].numpy()
        return embeddings

    def model_predict(self, X_test, model):
        """
        基于模型预测意图
        :param X_test: query的关键词array
        :param model:朴素贝叶斯模型
        :return:
        """
        print('EntityExtractor.model_predict()......')

        tokenizer = BertTokenizer.from_pretrained('./pretrained_bert_chinese/')
        model_bert = BertModel.from_pretrained('./pretrained_bert_chinese/')

        # # 冻结模型参数
        # for param in model_bert.parameters():
        #     param.requires_grad_(False)

        # # python训练朴素贝叶斯模型,并保存训练好的模型
        # X = [
        #      '什么是糖尿病', '请告诉我更多关于高血压的信息', '艾滋病的基本知识是什么', '肝肿大是什么病', '什么是高血压？', '糖尿病的原因是什么？', '癌症有哪些类型？', '什么是哮喘？', '冠心病的原因是什么？', '类风湿关节炎有哪些类型？', '什么是胃溃疡？', '抑郁症的原因是什么？', '甲状腺功能亢进有哪些类型？', '什么是胃溃疡？', '帕金森病的原因是什么？', '慢性支气管炎有哪些类型？', '什么是肾结石？', '阿尔茨海默病的原因是什么？', '红斑狼疮有哪些类型？',
        #      '糖尿病有哪些症状', '发烧时会有哪些表现', '皮疹是什么症状', '慢性乙肝有什么表现', '高血压有什么症状？', '糖尿病的早期症状是什么？', '癌症的常见症状有哪些？', '哮喘有什么症状？', '冠心病的早期症状是什么？', '类风湿关节炎的常见症状有哪些？', '胃溃疡有什么症状？', '抑郁症的早期症状是什么？', '甲状腺功能亢进的常见症状有哪些？', '胃溃疡有什么症状？', '帕金森病的早期症状是什么？', '慢性支气管炎的常见症状有哪些？', '肾结石有什么症状？', '阿尔茨海默病的早期症状是什么？', '红斑狼疮的常见症状有哪些？',
        #      '糖尿病的治疗方法有哪些', '怎样治疗高血压', '癌症的常见治疗方案有哪些', '肚子一直痛怎么办', '高血压应该怎么治疗？', '糖尿病的治疗方法有哪些？', '癌症有哪些治疗方案？', '哮喘应该怎么治疗？', '冠心病的治疗方法有哪些？', '类风湿关节炎有哪些治疗方案？', '胃溃疡应该怎么治疗？', '抑郁症的治疗方法有哪些？', '甲状腺功能亢进有哪些治疗方案？', '胃溃疡应该怎么治疗？', '帕金森病的治疗方法有哪些？', '慢性支气管炎有哪些治疗方案？', '肾结石应该怎么治疗？', '阿尔茨海默病的治疗方法有哪些？', '红斑狼疮有哪些治疗方案？',
        #      '确诊糖尿病需要做哪些检查', '高血压需要做什么检查', '肺炎的检查项目有哪些', '癌症筛查需要做哪些测试', '高血压需要做哪些检查？', '糖尿病的诊断需要做哪些检查？', '癌症的确诊需要进行哪些检查？', '哮喘需要做哪些检查？', '冠心病的诊断需要做哪些检查？', '类风湿关节炎的确诊需要进行哪些检查？', '胃溃疡需要做哪些检查？', '抑郁症的诊断需要做哪些检查？', '甲状腺功能亢进的确诊需要进行哪些检查？', '胃溃疡需要做哪些检查？', '帕金森病的诊断需要做哪些检查？', '慢性支气管炎的确诊需要进行哪些检查？', '肾结石需要做哪些检查？', '阿尔茨海默病的诊断需要做哪些检查？', '红斑狼疮的确诊需要进行哪些检查？',
        #      '糖尿病应该看哪个科', '皮肤病要挂什么科', '心脏问题要去哪个科室', '怀疑自己有肾病该看哪个科', '高血压应该去哪个科室看？', '糖尿病应该挂什么科？', '得了癌症应该去哪个科室就诊？', '哮喘应该去哪个科室看？', '冠心病应该挂什么科？', '得了类风湿关节炎应该去哪个科室就诊？', '胃溃疡应该去哪个科室看？', '抑郁症应该挂什么科？', '得了甲状腺功能亢进应该去哪个科室就诊？', '胃溃疡应该去哪个科室看？', '帕金森病应该挂什么科？', '得了慢性支气管炎应该去哪个科室就诊？', '肾结石应该去哪个科室看？', '阿尔茨海默病应该挂什么科？', '得了红斑狼疮应该去哪个科室就诊？',
        #      '糖尿病的治愈率高吗', '癌症的治愈率是多少', '高血压能治愈吗', '肺炎的治愈率如何', '高血压的治愈率是多少？', '糖尿病的治愈率高吗？', '癌症的治愈率有多高？', '哮喘的治愈率是多少？', '冠心病的治愈率高吗？', '类风湿关节炎的治愈率有多高？', '胃溃疡的治愈率是多少？', '抑郁症的治愈率高吗？', '甲状腺功能亢进的治愈率有多高？', '胃溃疡的治愈率是多少？', '帕金森病的治愈率高吗？', '慢性支气管炎的治愈率有多高？', '肾结石的治愈率是多少？', '阿尔茨海默病的治愈率高吗？', '红斑狼疮的治愈率有多高？',
        #      '乙肝多久能治好', '高血压需要多长时间能控制', '治愈癌症需要多长时间', '肾结石治愈周期是多长', '高血压需要多长时间能治好？', '糖尿病需要治疗多久？', '癌症的治愈周期是多长？', '哮喘需要多长时间能治好？', '冠心病需要治疗多久？', '类风湿关节炎的治愈周期是多长？', '胃溃疡需要多长时间能治好？', '抑郁症需要治疗多久？', '甲状腺功能亢进的治愈周期是多长？', '胃溃疡需要多长时间能治好？', '帕金森病需要治疗多久？', '慢性支气管炎的治愈周期是多长？', '肾结石需要多长时间能治好？', '阿尔茨海默病需要治疗多久？', '红斑狼疮的治愈周期是多长？',
        #      '请告诉我糖尿病的所有信息', '关于高血压的全部知识是什么', '详细介绍一下艾滋病', '慢性咽炎', '高血压的所有相关信息是什么？', '糖尿病的全面介绍是什么？', '癌症的详细信息包括哪些？', '哮喘的所有相关信息是什么？', '冠心病的全面介绍是什么？', '类风湿关节炎的详细信息包括哪些？', '肺炎的所有相关信息是什么？', '抑郁症的全面介绍是什么？', '甲状腺功能亢进的详细信息包括哪些？', '胃溃疡的所有相关信息是什么？', '帕金森病的全面介绍是什么？', '慢性支气管炎的详细信息包括哪些？', '肾结石的所有相关信息是什么？', '阿尔茨海默病的全面介绍是什么？', '红斑狼疮的详细信息包括哪些？'
        #      ]
        # X_train = []
        # X_test = []
        # Y_train = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        #            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        #            2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
        #            3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
        #            4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
        #            5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
        #            6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
        #            7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7]
        #
        # # 构建每个query的embedding
        # for i in range(len(X)):
        #     inputs = tokenizer(X[i], return_tensors='pt', max_length=512, truncation=True, padding='max_length')
        #     with torch.no_grad():
        #         outputs = model_bert(**inputs)
        #     sentence_vector = outputs.last_hidden_state[:, 0, :].squeeze()
        #     X_train.append(sentence_vector.tolist())  # 获取句子向量（通常使用[CLS]标记的输出作为句子向量）
        #
        # # MultinomialNB(朴素贝叶斯)要求训练集中不出现负值
        # sclar = MinMaxScaler()
        # sclar.fit(X_train)
        # X_train_sclar = sclar.transform(X_train)
        # model.fit(X_train_sclar, Y_train)
        #
        # outputs_test = tokenizer(' '.join([str(i) for i in X_test]), return_tensors='pt', max_length=512, truncation=True, padding='max_length')

        X_test = [str(i) for i in X_test]
        X_test, test_masks = self.encode_texts([' '.join(X_test)], tokenizer)
        test_embeddings = self.get_bert_embeddings(X_test, test_masks, model_bert)

        res = model.predict(test_embeddings)
        labels = ['query_disease', 'query_symptom', 'query_cureway', 'query_checklist', 'query_department', 'query_rate', 'query_period', 'disease_describe']

        return labels[res[0]]

    def extractor(self, question):
        """
        实体抽取主函数
        :param question: 用户给出的query
        :return:
        """
        print('EntityExtractor.extractor()......')

        # 模式匹配, 得到匹配的词和类型。self.result={'Disease':疾病名, 'Alias':疾病别名, 'Complication':并发症, 'Symptom':症状}
        self.entity_reg(question)
        if not self.result:
            self.find_sim_words(question)  # 当全匹配失败时，就采用相似度计算来找相似的词

        # 记录实体类型
        types = []
        for v in self.result.keys():
            types.append(v)

        # 记录查询意图
        intentions = []
        tfidf_feature = self.tfidf_features(question, self.tfidf_model)

        if tfidf_feature[0] != '':
            # other_feature = self.other_features(question)
            # m = other_feature.shape
            # other_feature = np.reshape(other_feature, (1, m[0]))
            # feature = np.concatenate((tfidf_feature, other_feature), axis=1)
            feature = np.concatenate((tfidf_feature, self.other_features(question)), axis=0)

            predicted = self.model_predict(feature, self.nb_model)  # 1.基于模型预测意图
            # intentions.append(predicted[0])
            intentions.append(predicted)

        # 2.基于特征词分类意图（处理多意图）
        # 已知“疾病/别名”，查询症状
        # self.check_words(self.symptom_qwds, question)：判断self.symptom_qwds中是否有对意图的特征词在question中
        if self.check_words(self.symptom_qwds, question) and ('Disease' in types or 'Alia' in types):
            intention = "query_symptom"
            if intention not in intentions:
                intentions.append(intention)

        # 已知“疾病/别名”or“症状/并发症”，查询治疗方法
        if self.check_words(self.cureway_qwds, question) and ('Disease' in types or 'Symptom' in types or 'Alias' in types or 'Complication' in types):
            intention = "query_cureway"
            if intention not in intentions:
                intentions.append(intention)

        # 已知“疾病/别名”，查询治疗周期
        if self.check_words(self.lasttime_qwds, question) and ('Disease' in types or 'Alia' in types):
            intention = "query_period"
            if intention not in intentions:
                intentions.append(intention)

        # 已知“疾病/别名”，查询治愈率
        if self.check_words(self.cureprob_qwds, question) and ('Disease' in types or 'Alias' in types):
            intention = "query_rate"
            if intention not in intentions:
                intentions.append(intention)

        # 已知“疾病/别名”，查询检查项目
        if self.check_words(self.check_qwds, question) and ('Disease' in types or 'Alias' in types):
            intention = "query_checklist"
            if intention not in intentions:
                intentions.append(intention)

        # 已知“疾病/别名”or“症状/并发症”，查询科室
        if self.check_words(self.belong_qwds, question) and ('Disease' in types or 'Symptom' in types or 'Alias' in types or 'Complication' in types):
            intention = "query_department"
            if intention not in intentions:
                intentions.append(intention)

        # 已知“症状/并发症”，查询疾病
        if self.check_words(self.disase_qwds, question) and ("Symptom" in types or "Complication" in types):
            intention = "query_disease"
            if intention not in intentions:
                intentions.append(intention)

        # 若没有检测到意图，且已知疾病，则返回疾病的描述
        if not intentions and ('Disease' in types or 'Alias' in types):
            intention = "disease_describe"
            if intention not in intentions:
                intentions.append(intention)

        # 若是疾病和症状同时出现，且出现了查询疾病的特征词，则意图为查询疾病
        if self.check_words(self.disase_qwds, question) and ('Disease' in types or 'Alias' in types) and ("Symptom" in types or "Complication" in types):
            intention = "query_disease"
            if intention not in intentions:
                intentions.append(intention)

        # 若没有识别出实体或意图则调用其它方法
        if not intentions or not types:
            intention = "QA_matching"
            if intention not in intentions:
                intentions.append(intention)

        self.result["intentions"] = intentions

        return self.result


if __name__ == "__main__":

    # # 常见问题测试：单意图
    # print(EntityExtractor().extractor('我最近总头晕是怎么回事？'))  # 0(query_disease)
    # print(EntityExtractor().extractor('地中海拼写的治愈率高吗？'))  # 5(query_rate)
    # print(EntityExtractor().extractor('地中海贫血有什么症状？'))  # 1(query_symptom)
    # print(EntityExtractor().extractor('地中海贫血怎么治疗？'))  # 2(query_cureway)
    # print(EntityExtractor().extractor('地中海贫血挂什么科？'))  # 4(query_department)   -------------------------
    # print(EntityExtractor().extractor('地中海贫血多久能治好？'))  # 6(query_period)
    # print(EntityExtractor().extractor('详细介绍一下地中海贫血？'))  # 7(disease_describe)
    # print(EntityExtractor().extractor('地中海贫血需要做哪些检查？'))  # 3(query_checklist)

    # 常见问题测试：多意图
    # print(EntityExtractor().extractor('地中海贫血有什么症状？多长时间能治好？'))  # 0(query_disease), 6(query_period)
    # print(EntityExtractor().extractor('地中海贫血的治疗方法有哪些?可以治吗?'))  # 2(query_cureway), 5(query_rate)
    # print(EntityExtractor().extractor('地中海贫血有什么症状？可以治吗?在医院需要挂什么科？'))  # 0(query_disease) 5(query_rate) 4(query_department)

    # 极端问题测试
    # print(EntityExtractor().extractor('你是谁？'))
    # print(EntityExtractor().extractor('你真是个大傻叉！'))
    # print(EntityExtractor().extractor('转人工！'))
    # print(EntityExtractor().extractor('&……￥#*%￥！'))
    print(EntityExtractor().extractor('昨天发烧，服用了阿司匹林,并且还吃了牛黄清胃丸，饭是吃了瓜烧白菜，大便有点色浅，可能是什么病？'))

