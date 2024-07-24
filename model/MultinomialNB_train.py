import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from transformers import BertTokenizer, BertModel
import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split



# 加载预训练的BERT模型和分词器
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = BertModel.from_pretrained('bert-base-chinese')


# 将文本转化为BERT输入格式
def encode_texts(texts, tokenizer, max_length=128):
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


# 获取BERT向量表示
def get_bert_embeddings(inputs, masks, model):
    with torch.no_grad():
        outputs = model(inputs, attention_mask=masks)
        hidden_states = outputs.last_hidden_state
        # 使用句子的[CLS] token表示
        embeddings = hidden_states[:, 0, :].numpy()
    return embeddings


def train(train_inputs, train_labels):
    train_embeddings = get_bert_embeddings(train_inputs, train_masks, model)

    sclar = MinMaxScaler()
    sclar.fit(train_embeddings)
    train_embeddings_sclar = sclar.transform(train_embeddings)

    # 训练朴素贝叶斯分类器
    nb_classifier = MultinomialNB()
    nb_classifier.fit(train_embeddings_sclar, train_labels)

    # 保存模型
    joblib.dump(nb_classifier, 'nb_classifier.joblib')


if __name__ == "__main__":
    # 假设你的数据如下（示例数据）
    data = {'text':
                ['什么是糖尿病', '请告诉我更多关于高血压的信息', '艾滋病的基本知识是什么', '肝肿大是什么病', '什么是高血压？', '糖尿病的原因是什么？',
                 '癌症有哪些类型？', '什么是哮喘？', '冠心病的原因是什么？', '类风湿关节炎有哪些类型？', '什么是胃溃疡？', '抑郁症的原因是什么？',
                 '甲状腺功能亢进有哪些类型？', '什么是胃溃疡？', '帕金森病的原因是什么？', '慢性支气管炎有哪些类型？', '什么是肾结石？', '阿尔茨海默病的原因是什么？',
                 '红斑狼疮有哪些类型？', '我最近经常感觉到胸口疼痛，这可能是心绞痛吗？', '我爷爷最近常常忘记事情，这是老年痴呆症的早期症状吗？', '我手指关节肿胀疼痛，这可能是类风湿关节炎吗？',
                 '我最近老是感觉肚子疼，可能是胃溃疡吗？', '我听说支气管炎很常见，能告诉我它具体是什么吗？', '我妈妈被诊断出有多发性硬化症，这是什么病啊？',
                 '我最近听说过幽门螺杆菌，这是什么细菌，会引起什么疾病吗？', '我的朋友被诊断出有银屑病，这是一种什么样的病？', '医生说我可能有克罗恩病，能解释一下这是什么病吗？',

                 '糖尿病有哪些症状', '发烧时会有哪些表现', '皮疹是什么症状', '慢性乙肝有什么表现', '高血压有什么症状？','糖尿病的早期症状是什么？',
                 '癌症的常见症状有哪些？', '哮喘有什么症状？', '冠心病的早期症状是什么？', '类风湿关节炎的常见症状有哪些？', '胃溃疡有什么症状？',
                 '抑郁症的早期症状是什么？', '甲状腺功能亢进的常见症状有哪些？', '胃溃疡有什么症状？', '帕金森病的早期症状是什么？', '慢性支气管炎的常见症状有哪些？',
                 '肾结石有什么症状？', '阿尔茨海默病的早期症状是什么？', '红斑狼疮的常见症状有哪些？', '最近我总是感觉气短，可能是什么原因？', '我早上起床时手脚麻木，这是什么症状？', '我胃部经常不舒服，感觉有一块硬块，这可能是什么问题？',
                 '我经常觉得头晕、恶心，这是什么症状？', '最近嗓子总是痛，可能是什么原因？', '我的皮肤上出现了红色斑点，伴有瘙痒，这是什么症状？', '我总是觉得胸口灼热，这是胃食管反流病的症状吗？',
                 '我的关节总是僵硬和疼痛，这是风湿病的症状吗？', '最近老是觉得嗜睡和没有食欲，这可能是什么病的症状？',

                 '糖尿病的治疗方法有哪些', '怎样治疗高血压', '癌症的常见治疗方案有哪些', '肚子一直痛怎么办', '高血压应该怎么治疗？', '糖尿病的治疗方法有哪些？',
                 '癌症有哪些治疗方案？', '哮喘应该怎么治疗？', '冠心病的治疗方法有哪些？', '类风湿关节炎有哪些治疗方案？', '胃溃疡应该怎么治疗？', '抑郁症的治疗方法有哪些？',
                 '甲状腺功能亢进有哪些治疗方案？', '胃溃疡应该怎么治疗？', '帕金森病的治疗方法有哪些？', '慢性支气管炎有哪些治疗方案？', '肾结石应该怎么治疗？',
                 '阿尔茨海默病的治疗方法有哪些？', '红斑狼疮有哪些治疗方案？', '我被诊断出高血压，有什么自然疗法可以控制血压？', '我得了慢性肾病，除了药物治疗外，还有什么其他方法可以帮助？',
                 '我最近被诊断出慢性胃炎，有什么饮食上的建议可以减轻症状？', '医生说我有胆结石，治疗起来痛苦吗？', '我得了慢性咽炎，有什么有效的治疗方法吗？', '皮肤癣怎么治才能彻底好？',
                 '幽门螺杆菌感染该怎么治疗？', '银屑病的治疗方法有哪些，有没有什么新的疗法？', '克罗恩病有治愈的方法吗？',

                 '确诊糖尿病需要做哪些检查', '高血压需要做什么检查', '肺炎的检查项目有哪些', '癌症筛查需要做哪些测试', '高血压需要做哪些检查？',
                 '糖尿病的诊断需要做哪些检查？', '癌症的确诊需要进行哪些检查？', '哮喘需要做哪些检查？', '冠心病的诊断需要做哪些检查？', '类风湿关节炎的确诊需要进行哪些检查？',
                 '胃溃疡需要做哪些检查？', '抑郁症的诊断需要做哪些检查？', '甲状腺功能亢进的确诊需要进行哪些检查？', '胃溃疡需要做哪些检查？', '帕金森病的诊断需要做哪些检查？',
                 '慢性支气管炎的确诊需要进行哪些检查？', '肾结石需要做哪些检查？', '阿尔茨海默病的诊断需要做哪些检查？', '红斑狼疮的确诊需要进行哪些检查？',
                 '我最近出现头痛和视力模糊，需要做哪些检查来排除脑部问题？', '我经常感到疲劳和心跳过快，需要进行哪些检查来检查心脏健康？', '我有家族遗传病史，需要定期做哪些检查来预防？',
                 '我咳嗽了好几周了，应该做哪些检查？', '我体重突然下降，需要做哪些检查来找出原因？', '我眼睛最近视力下降，需要做什么检查？', '我经常胃痛，应该做哪些检查来确定是不是胃溃疡？',
                 '医生建议我做结肠镜检查，这对诊断克罗恩病有帮助吗？', '银屑病需要做哪些检查来确诊？',

                 '糖尿病应该看哪个科', '皮肤病要挂什么科', '心脏问题要去哪个科室', '怀疑自己有肾病该看哪个科', '高血压应该去哪个科室看？', '糖尿病应该挂什么科？',
                 '得了癌症应该去哪个科室就诊？', '哮喘应该去哪个科室看？', '冠心病应该挂什么科？', '得了类风湿关节炎应该去哪个科室就诊？', '胃溃疡应该去哪个科室看？',
                 '抑郁症应该挂什么科？', '得了甲状腺功能亢进应该去哪个科室就诊？', '胃溃疡应该去哪个科室看？', '帕金森病应该挂什么科？', '得了慢性支气管炎应该去哪个科室就诊？',
                 '肾结石应该去哪个科室看？', '阿尔茨海默病应该挂什么科？', '得了红斑狼疮应该去哪个科室就诊？', '我有胃部不适和消化问题，应该去哪个科室就诊？',
                 '我有心脏方面的问题，应该挂心脏科还是内科？', '我怀疑自己有精神健康问题，应该去精神科还是神经科？', '我有长期的背痛，应该去骨科还是康复科？', '皮肤上长了奇怪的疹子，该看皮肤科还是传染病科？',
                 '我的小孩老是咳嗽，去儿科还是呼吸科好？', '胃食管反流病应该看消化内科还是内科？', '风湿病要去风湿免疫科吗？', '我的皮肤上出现很多鳞屑，应该挂皮肤科吗？',

                 '糖尿病的治愈率高吗', '癌症的治愈率是多少', '高血压能治愈吗', '肺炎的治愈率如何', '高血压的治愈率是多少？', '糖尿病的治愈率高吗？', '癌症的治愈率有多高？',
                 '哮喘的治愈率是多少？', '冠心病的治愈率高吗？', '类风湿关节炎的治愈率有多高？', '胃溃疡的治愈率是多少？', '抑郁症的治愈率高吗？', '甲状腺功能亢进的治愈率有多高？',
                 '胃溃疡的治愈率是多少？', '帕金森病的治愈率高吗？', '慢性支气管炎的治愈率有多高？', '肾结石的治愈率是多少？', '阿尔茨海默病的治愈率高吗？', '红斑狼疮的治愈率有多高？',
                 '治疗白血病的成功率是多少？', '治疗乳腺癌的成功率有多高？', '治疗糖尿病的成功率和长期控制效果如何？', '治好胃癌的几率有多大？', '白内障手术的成功率高吗？', '肺炎治愈的概率是多少？',
                 '胃食管反流病能彻底治好吗？', '克罗恩病的治愈率有多高？', '风湿病的治愈率是多少？',

                 '乙肝多久能治好', '高血压需要多长时间能控制', '治愈癌症需要多长时间', '肾结石治愈周期是多长', '高血压需要多长时间能治好？', '糖尿病需要治疗多久？',
                 '癌症的治愈周期是多长？', '哮喘需要多长时间能治好？', '冠心病需要治疗多久？', '类风湿关节炎的治愈周期是多长？', '胃溃疡需要多长时间能治好？', '抑郁症需要治疗多久？',
                 '甲状腺功能亢进的治愈周期是多长？', '胃溃疡需要多长时间能治好？', '帕金森病需要治疗多久？', '慢性支气管炎的治愈周期是多长？', '肾结石需要多长时间能治好？',
                 '阿尔茨海默病需要治疗多久？', '红斑狼疮的治愈周期是多长？', '治疗甲状腺问题通常需要多长时间？', '治疗骨折后的康复过程会持续多久？', '治疗哮喘能够见效需要多久时间？',
                 '治疗急性阑尾炎需要多长时间？', '我有带状疱疹，需要多长时间才能好？', '肾结石需要多长时间才能排出？', '治疗幽门螺杆菌需要多长时间？', '银屑病的治疗周期一般是多久？',
                 '风湿病的治疗周期是多长？',

                 '请告诉我糖尿病的所有信息', '关于高血压的全部知识是什么', '详细介绍一下艾滋病', '慢性咽炎', '高血压的所有相关信息是什么？', '糖尿病的全面介绍是什么？',
                 '癌症的详细信息包括哪些？', '哮喘的所有相关信息是什么？', '冠心病的全面介绍是什么？', '类风湿关节炎的详细信息包括哪些？', '肺炎的所有相关信息是什么？',
                 '抑郁症的全面介绍是什么？', '甲状腺功能亢进的详细信息包括哪些？', '胃溃疡的所有相关信息是什么？', '帕金森病的全面介绍是什么？', '慢性支气管炎的详细信息包括哪些？',
                 '肾结石的所有相关信息是什么？', '阿尔茨海默病的全面介绍是什么？', '红斑狼疮的详细信息包括哪些？', '我想了解更多关于糖尿病的信息，包括病因、症状、治疗方法等',
                 '关于癫痫病的全面介绍，包括常见症状和治疗选择', '我对帕金森病有兴趣，想了解该病的进展和预后。', '能给我讲讲关于痛风的所有细节吗？', '我想知道关于荨麻疹的所有信息，包括症状和预防。',
                 '能介绍一下子宫肌瘤的全面信息吗，包括病因和治疗方法？', '我想了解一下幽门螺杆菌感染的所有信息，包括传染途径和预防。', '能详细介绍一下银屑病吗，包括病因、症状和治疗？',
                 '克罗恩病的全面信息是什么？我需要知道它的症状、治疗方法和预后'
                 ],
            'label': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                      2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                      3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                      4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                      5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
                      6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
                      7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7]
            }

    df = pd.DataFrame(data)

    # 分割数据集为训练集和测试集
    train_texts, test_texts, train_labels, test_labels = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

    # 对训练集和测试集进行编码
    train_inputs, train_masks = encode_texts(train_texts, tokenizer)
    test_inputs, test_masks = encode_texts(test_texts, tokenizer)

    # 训练并保存模型
    train(train_inputs, train_labels)

    # 生成测试集embedding
    test_embeddings = get_bert_embeddings(test_inputs, test_masks, model)

    # 加载模型
    loaded_nb_classifier = joblib.load('nb_classifier.joblib')

    # 预测并评估分类器性能
    test_preds = loaded_nb_classifier.predict(test_embeddings)
    accuracy = accuracy_score(test_labels, test_preds)
    report = classification_report(test_labels, test_preds)

    print(test_preds)
    print(f'Accuracy: {accuracy}')
    print('Classification Report:')
    print(report)




# model = MultinomialNB()
#
# # 加载预训练模型
# tokenizer = BertTokenizer.from_pretrained('../pretrained_bert_chinese/')
# model_bert = BertModel.from_pretrained('../pretrained_bert_chinese/')
#
# # 冻结Bert模型参数
# for param in model_bert.parameters():
#     param.requires_grad_(False)
#
# X_train_bert = []
# X_test_bert = []
#
# # 构建X_train的embedding
# for i in range(len(X_train)):
#     inputs = tokenizer(X_train[i], return_tensors='pt', max_length=512, truncation=True, padding='max_length')
#     outputs = model_bert(**inputs)
#     # sentence_vector = outputs.last_hidden_state[:, 0, :].squeeze()  # 获取句子向量（通常使用[CLS]标记的输出作为句子向量）
#     sentence_vector = outputs.last_hidden_state
#     X_train_bert.append(sentence_vector.tolist())
#
# # MultinomialNB(朴素贝叶斯)要求训练集中不出现负值
# sclar = MinMaxScaler()
# sclar.fit(X_train)
# X_train_sclar = sclar.transform(X_train)
# model.fit(X_train_sclar, Y_train)
#
# outputs_test = model_bert(**tokenizer('什么是糖尿病', return_tensors='pt', max_length=512, truncation=True, padding='max_length')).last_hidden_state
# X_test_bert.append(outputs_test.tolist())
# res = model.predict(X_test_bert)
# labels = ['query_disease', 'query_symptom', 'query_cureway', 'query_checklist', 'query_department', 'query_rate', 'query_period', 'disease_describe']
#
# print(labels[res[0]])






# # 初始化TfidfVectorizer
# vectorizer = TfidfVectorizer(stop_words='english')
#
# # 转换训练
# X_train_tfidf = vectorizer.fit_transform(X_train)
#
# # 初始化朴素贝叶斯模型
# nb_model = MultinomialNB()
#
# # 训练模型
# nb_model.fit(X_train_tfidf, y_train)
#
# # 保存模型和向量化器
# joblib.dump(nb_model, 'naive_bayes_model.pkl')
# joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
#
# # 加载模型和向量化器
# nb_model_loaded = joblib.load('naive_bayes_model.pkl')
# vectorizer_loaded = joblib.load('tfidf_vectorizer.pkl')
#
# # 转换测试数据
# # X_test = ["I love programming in Python"]
# X_test = ["头疼的治愈率是多少"]
# X_test_loaded_tfidf = vectorizer_loaded.transform(X_test)
#
# # 进行预测
# predictions = nb_model_loaded.predict(X_test_loaded_tfidf)
#
# # 打印预测结果
# print(predictions)
