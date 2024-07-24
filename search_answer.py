# -*- coding: utf-8 -*-
# @Author : maxin-nk
# @Email : xxin_ma@163.com
# @File : search_answer
from py2neo import Graph
import re


class AnswerSearching:
    def __init__(self, top_num=10):
        self.graph = Graph("bolt://localhost:7687", auth=("neo4j", "571773795ma"), name="neo4j")
        self.top_num = top_num

    def question_parser(self, data):
        """
        主要是根据不同的实体和意图构造cypher查询语句
        :param data: {"Disease":[], "Alias":[], "Symptom":[], "Complication":[], "intentions":[]}
        :return: [{'intention':查询意图, 'sql':cypher查询语句}]
        """
        sqls = []
        if data:
            for intent in data["intentions"]:
                sql_ = {"intention": intent}
                sql = []
                if data.get("Disease"):
                   sql = self.transfor_to_sql("Disease", data["Disease"], intent)
                elif data.get("Alias"):
                    sql = self.transfor_to_sql("Alias", data["Alias"], intent)
                elif data.get("Symptom"):
                    sql = self.transfor_to_sql("Symptom", data["Symptom"], intent)
                elif data.get("Complication"):
                    sql = self.transfor_to_sql("Complication", data["Complication"], intent)

                if sql:
                    sql_['sql'] = sql
                    sqls.append(sql_)
        return sqls

    def transfor_to_sql(self, label, entities, intent):
        """
        将问题转变为cypher查询语句
        :param label:实体标签
        :param entities:实体列表
        :param intent:查询意图
        :return:cypher查询语句
        """
        if not entities:
            return []
        sql = []

        # 按照 intention 分类返回查询结果
        # 查询疾病（实体）
        if intent == "query_disease" and label == "Alias":  # 别名->疾病
            sql = ["MATCH (d:Disease)-[]->(s:Alias) WHERE s.name='{0}' return d.name".format(e) for e in entities]
        if intent == "query_disease" and label == "Symptom":  # 症状->疾病
            sql = ["MATCH (d:Disease)-[]->(s:Symptom) WHERE s.name='{0}' return d.name".format(e) for e in entities]
        # if intent == "query_disease" and label == "Complication":  # 并发症->疾病（自建）
        #     sql = ["MATCH (d:Disease)-[]->(s:Complication) WHERE s.name='{0}' return d.name".format(e) for e in entities]

        # 查询症状（实体）
        if intent == "query_symptom" and label == "Disease":  # 疾病->症状
            sql = ["MATCH (d:Disease)-[:HAS_SYMPTOM]->(s) WHERE d.name='{0}' RETURN d.name,s.name".format(e) for e in entities]  # 查找所有从标签为 Disease 的节点 d 出发，通过 HAS_SYMPTOM 关系连接到标签为 s 的节点。
        if intent == "query_symptom" and label == "Alias":  # 别名->症状
            sql = ["MATCH (a:Alias)<-[:ALIAS_IS]-(d:Disease)-[:HAS_SYMPTOM]->(s) WHERE a.name='{0}' return d.name,s.name".format(e) for e in entities]

        # 查询治疗方法（实体）
        if intent == "query_cureway" and label == "Disease":  # 疾病->治疗方法
            sql = ["MATCH (d:Disease)-[:HAS_DRUG]->(n) WHERE d.name='{0}' return d.name,d.treatment, n.name".format(e) for e in entities]
        if intent == "query_cureway" and label == "Alias":  # 别名->治疗方法
            sql = ["MATCH (n)<-[:HAS_DRUG]-(d:Disease)-[]->(a:Alias) WHERE a.name='{0}' return d.name, d.treatment, n.name".format(e) for e in entities]
        if intent == "query_cureway" and label == "Symptom":  # 症状->治疗方法
            sql = ["MATCH (n)<-[:HAS_DRUG]-(d:Disease)-[]->(s:Symptom) WHERE s.name='{0}' return d.name,d.treatment, n.name".format(e) for e in entities]
        if intent == "query_cureway" and label == "Complication":  # 并发症->治疗方法
            sql = ["MATCH (n)<-[:HAS_DRUG]-(d:Disease)-[]->(c:Complication) WHERE c.name='{0}' return d.name,d.treatment, n.name".format(e) for e in entities]

        # 查询治疗周期（属性）
        if intent == "query_period" and label == "Disease":  # 疾病->治疗周期
            sql = ["MATCH (d:Disease) WHERE d.name='{0}' return d.name,d.period".format(e) for e in entities]
        if intent == "query_period" and label == "Alias":  # 别名->治疗周期
            sql = ["MATCH (d:Disease)-[]->(a:Alias) WHERE a.name='{0}' return d.name,d.period".format(e) for e in entities]
        if intent == "query_period" and label == "Symptom":  # 症状->治疗周期
            sql = ["MATCH (d:Disease)-[]->(s:Symptom) WHERE s.name='{0}' return d.name,d.period".format(e) for e in entities]
        if intent == "query_period" and label == "Complication":  # 并发症->治疗周期
            sql = ["MATCH (d:Disease)-[]->(c:Complication) WHERE c.name='{0}' return d.name, d.period".format(e) for e in entities]

        # 查询治愈率（属性）
        if intent == "query_rate" and label == "Disease":  # 疾病->治愈率
            sql = ["MATCH (d:Disease) WHERE d.name='{0}' return d.name,d.rate".format(e) for e in entities]
        if intent == "query_rate" and label == "Alias":  # 别名->治愈率
            sql = ["MATCH (d:Disease)-[]->(a:Alias) WHERE a.name='{0}' return d.name,d.rate".format(e) for e in entities]
        if intent == "query_rate" and label == "Symptom":  # 症状->治愈率
            sql = ["MATCH (d:Disease)-[]->(s:Symptom) WHERE s.name='{0}' return d.name,d.rate".format(e) for e in entities]
        if intent == "query_rate" and label == "Complication":  # 并发症->治愈率
            sql = ["MATCH (d:Disease)-[]->(c:Complication) WHERE c.name='{0}' return d.name, d.rate".format(e) for e in entities]

        # 查询检查项目（属性）
        if intent == "query_checklist" and label == "Disease":  # 疾病->检查项目
            sql = ["MATCH (d:Disease) WHERE d.name='{0}' return d.name,d.checklist".format(e) for e in entities]
        if intent == "query_checklist" and label == "Alias":  # 别名->检查项目
            sql = ["MATCH (d:Disease)-[]->(a:Alias) WHERE a.name='{0}' return d.name,d.checklist".format(e) for e in entities]
        if intent == "query_checklist" and label == "Symptom":  # 症状->检查项目
            sql = ["MATCH (d:Disease)-[]->(s:Symptom) WHERE s.name='{0}' return d.name, d.checklist".format(e) for e in entities]
        if intent == "query_checklist" and label == "Complication":  # 并发症->检查项目
            sql = ["MATCH (d:Disease)-[]->(c:Complication) WHERE c.name='{0}' return d.name, d.checklist".format(e) for e in entities]

        # 查询科室（属性）
        if intent == "query_department" and label == "Disease":  # 疾病->科室
            sql = ["MATCH (d:Disease)-[:DEPARTMENT_IS]->(n) WHERE d.name='{0}' return d.name, n.name".format(e) for e in entities]
        if intent == "query_department" and label == "Alias":  # 别名->科室
            sql = ["MATCH (n)<-[:DEPARTMENT_IS]-(d:Disease)-[:ALIAS_IS]->(a:Alias) WHERE a.name='{0}' return d.name,n.name".format(e) for e in entities]
        if intent == "query_department" and label == "Symptom":  # 症状->科室
            sql = ["MATCH (n)<-[:DEPARTMENT_IS]-(d:Disease)-[:HAS_SYMPTOM]->(s:Symptom) WHERE s.name='{0}' return d.name,n.name".format(e) for e in entities]
        if intent == "query_department" and label == "Complication":  # 并发症->科室
            sql = ["MATCH (n)<-[:DEPARTMENT_IS]-(d:Disease)-[:HAS_COMPLICATION]->(c:Complication) WHERE c.name='{0}' return d.name,n.name".format(e) for e in entities]

        # 查询疾病描述（属性）
        if intent == "disease_describe" and label == "Disease":  # 疾病->疾病描述
            sql = ["MATCH (d:Disease) WHERE d.name='{0}' return d.name,d.age,d.insurance,d.infection, d.checklist,d.period,d.rate,d.money".format(e) for e in entities]
        if intent == "disease_describe" and label == "Alias":  # 别名->疾病描述
            sql = ["MATCH (d:Disease)-[]->(a:Alias) WHERE a.name='{0}' return d.name,d.age, d.insurance,d.infection,d.checklist,d.period,d.rate,d.money".format(e) for e in entities]
        if intent == "disease_describe" and label == "Symptom":  # 症状->疾病描述
            sql = ["MATCH (d:Disease)-[]->(s:Symptom) WHERE s.name='{0}' return d.name,d.age, d.insurance,d.infection,d.checklist,d.period,d.rate,d.money".format(e) for e in entities]
        if intent == "disease_describe" and label == "Complication":  # 并发症->疾病描述
            sql = ["MATCH (d:Diosease)-[]->(c:Complicatin) WHERE c.name='{0}' return d.name, d.age,d.insurance,d.infection,d.checklist,d.period,d.rate,d.money".format(e) for e in entities]

        return sql

    def searching(self, sqls):
        """
        执行cypher查询，返回结果
        :param sqls: [{'intention':查询意图, 'sql':['cypher查询语句', ..]}, {}, ...]
        :return:str
        """
        final_answers = []
        for sql_ in sqls:
            intent = sql_['intention']
            queries = sql_['sql']
            answers = []

            # 编列当前查询意图下的多个cypher查询语句
            for query in queries:
                ress = self.graph.run(query).data()
                answers += ress  # ==answers.append

            # 将返回的查询结果格式化
            final_answer = self.answer_template(intent, answers)
            if final_answer:
                final_answers.append(final_answer)
        return final_answers

    def answer_template(self, intent, answers):
        """
        根据不同意图，返回不同模板的答案
        :param intent: 查询意图
        :param answers: 知识图谱查询结果
        :return: str
        """
        final_answer = ""
        if not answers:
            return ""

        # 查询疾病
        if intent == "query_disease":

            # 统计疾病出现的次数
            disease_freq = {}
            for data in answers:
                d = data["d.name"]
                disease_freq[d] = disease_freq.get(d, 0) + 1  # 如果key在字典中，则返回key的值，否则为默认值
            n = len(disease_freq.keys())

            # disease_freq.items():dict_items([('妊娠合并缺铁性贫血', 1), ('左心房心律', 2), ('光气中毒', 2), ('血虚眩晕', 2), ('Turcot综合征', 2), ('白细胞减少症', 3), ('胰岛功能性β细胞瘤', 1), ('自身免疫性溶血性贫血', 1), ('子痫', 2), ('氯丙嗪类中毒', 2), ('老年人慢性肾功能衰竭', 1), ('低血压', 4)])
            # sorted:按items中的值降序排列
            freq = sorted(disease_freq.items(), key=lambda x: x[1], reverse=True)

            for d, v in freq[:self.top_num]:
                final_answer += "疾病为 {0} 的概率为：{1}\n".format(d, v / len(freq[:self.top_num]))

        # 查询症状
        if intent == "query_symptom":
            disease_dic = {}
            for data in answers:
                d = data['d.name']
                s = data['s.name']
                if d not in disease_dic:
                    disease_dic[d] = [s]
                else:
                    disease_dic[d].append(s)  # 往字典value中添加症状

            # dict_items([('地中海贫血', ['骨质疏松', '脾肿大', '皮肤呈浅黄或深金黄色', '气血不足', '头晕', '重度贫血', '骨质疏松', '脾肿大', '皮肤呈浅黄或深金黄色', ...]), (), ...])
            i = 0
            for k, v in disease_dic.items():
                if i >= self.top_num:
                    break
                final_answer += "疾病 {0} 的症状包括：{1}\n".format(k, ','.join(list(set(v))))
                i += 1

        # 查询治疗方法
        if intent == "query_cureway":
            disease_dic = {}
            for data in answers:
                disease = data['d.name']

                # 处理treat冗余的问题
                treat = re.split(r'[,，.、\s+]', data["d.treatment"].rstrip())
                treat = [i for i in treat if len(i) >= 2]

                drug = data["n.name"]
                if disease not in disease_dic:
                    disease_dic[disease] = [treat, drug]
                else:
                    disease_dic[disease].append(drug)
                    disease_dic[disease][0].extend(treat)

            i = 0
            for d, v in disease_dic.items():
                if i >= self.top_num:
                    break

                # treatment去重
                final_answer += "疾病 {0} 的治疗方法有：{1}；通常使用的药品包括：{2}\n".format(d, set(v[0]), ','.join(v[1:]))
                i += 1

        # 查询治愈周期
        if intent == "query_period":
            disease_dic = {}
            for data in answers:
                d = data['d.name']
                p = data['d.period']
                if d not in disease_dic:
                    disease_dic[d] = [p]
                else:
                    disease_dic[d].append(p)

            i = 0
            for k, v in disease_dic.items():
                if i >= self.top_num:
                    break
                final_answer += "疾病 {0} 的治愈周期为：{1}\n".format(k, ','.join(list(set(v))))
                i += 1

        # 查询治愈率
        if intent == "query_rate":
            disease_dic = {}
            for data in answers:
                d = data['d.name']
                r = data['d.rate']
                if d not in disease_dic:
                    disease_dic[d] = [r]
                else:
                    disease_dic[d].append(r)

            i = 0
            for k, v in disease_dic.items():
                if i >= self.top_num:
                    break
                final_answer += "疾病 {0} 的治愈率为：{1}\n".format(k, ','.join(list(set(v))))
                i += 1

        # 查询检查项目
        if intent == "query_checklist":
            disease_dic = {}
            for data in answers:
                d = data['d.name']
                r = data['d.checklist']
                if d not in disease_dic:
                    disease_dic[d] = [r]
                else:
                    disease_dic[d].append(r)

            i = 0
            for k, v in disease_dic.items():
                if i >= self.top_num:
                    break
                final_answer += "疾病 {0} 通常可通过以下方检查确诊：{1}\n".format(k, ','.join(list(set(v))))
                i += 1

        # 查询科室
        if intent == "query_department":
            disease_dic = {}
            for data in answers:
                d = data['d.name']
                r = data['n.name']
                if d not in disease_dic:
                    disease_dic[d] = [r]
                else:
                    disease_dic[d].append(r)
            i = 0
            for k, v in disease_dic.items():
                if i >= self.top_num:
                    break
                final_answer += "疾病 {0} 可去以下科室就诊：{1}\n".format(k, ','.join(list(set(v))))
                i += 1

        # 查询疾病描述
        if intent == "disease_describe":
            disease_infos = {}
            for data in answers:
                name = data['d.name']
                age = data['d.age']
                insurance = data['d.insurance']
                infection = data['d.infection']
                checklist = data['d.checklist']
                period = data['d.period']
                rate = data['d.rate']
                money = data['d.money']
                if name not in disease_infos:
                    disease_infos[name] = [age, insurance, infection, checklist, period, rate, money]
                else:
                    disease_infos[name].extend([age, insurance, infection, checklist, period, rate, money])
            i = 0
            for k, v in disease_infos.items():
                if i >= 10:
                    break
                message = "疾病 {0} 的描述信息如下：\n发病人群：{1}\n医保：{2}\n传染性：{3}\n检查项目：{4}\n治愈周期：{5}\n治愈率：{6}\n费用：{7}\n"
                final_answer += message.format(k, v[0], v[1], v[2], v[3], v[4], v[5], v[6])
                i += 1

        return final_answer
