# -*- coding: utf-8 -*-
# @Author : maxin-nk
# @Email : xxin_ma@163.com
# @File : build_graph
from py2neo import Graph, Node
import pandas as pd
import re
import os


class MedicalGraph:
    def __init__(self):
        print("MedicalGraph.__init__()......")
        cur_dir = '\\'.join(os.path.abspath(__file__).split('\\')[:-1])
        self.data_path = os.path.join(cur_dir, 'data\\disease.csv')

        # 以下neo4j的连接方式不一定成功，主要取决于版本（可以逐一测试）
        # self.graph = Graph("http://localhost:7474", auth=("neo4j", "571773795ma"))
        # self.graph = Graph("bolt://localhost:7687", user="neo4j", password="571773795ma", name="neo4j")
        # self.graph = Graph("bolt://localhost:7687", auth=("neo4j", "571773795ma"))

        # 真实数据库
        self.graph = Graph("bolt://localhost:7687", auth=("neo4j", "571773795ma"), name="neo4j")

        # # 测试数据库
        # self.graph = Graph("bolt://localhost:7687", auth=("neo4j", "571773795ma"), name="test")

    def read_file(self):
        """
        读取文件，获得实体，实体之间关系，实体属性
        :return:返回实体集合,关系列表(二维),属性字典构成的列表
        """
        print("MedicalGraph.read_file()......")

        # 实体：Disease(疾病),Alias(别名),Symptom(症状),Part(发病部位),Department(所述科室),Complication(并发症),Drug(药品)
        diseases = []
        aliases = []
        symptoms = []
        parts = []
        departments = []
        complications = []
        drugs = []

        # 疾病的属性：age(发病人群), infection(是否传染), insurance(是否医保), checklist(检查项目), treatment(治疗方法), period(治愈周期), rate(治愈率), money(费用)
        diseases_infos = []

        # 关系
        disease_to_symptom = []  # 疾病与症状关系(HAS_SYMPTOM)
        disease_to_alias = []  # 疾病与别名关系(ALIAS_IS)
        diseases_to_part = []  # 疾病与部位关系(PART_IS)
        disease_to_department = []  # 疾病与科室关系(DEPARTMENT_IS)
        disease_to_complication = []  # 疾病与并发症关系(HAS_COMPLICATION)
        disease_to_drug = []  # 疾病与药品关系(HAS_DRUG)

        # 逐行处理数据
        # cols = ["name", "alias", "part", "age", "infection", "insurance", "department", "checklist", "symptom", "complication", "treatment", "drug", "period", "rate", "money"]
        # 数据类型：ndarray
        all_data = pd.read_csv(self.data_path, encoding='gb18030').loc[:, :].values
        for data in all_data:
            disease_dict = {}  # 以字典形式组织疾病属性

            # 实体-疾病(name)：放到属性字典构成的列表中了
            # str.strip():返回一个删除了前导和尾随空格的字符串副本。
            disease = str(data[0]).replace("...", " ").strip()
            disease_dict["name"] = disease

            # 实体-别名(alias):有存在多个别名的情况
            # [，、；,.;]:正则表达式，匹配[]中的任意字符
            line = re.sub("[，、；,.;]", " ", str(data[1])) if str(data[1]) else "未知"
            for alias in line.strip().split():
                aliases.append(alias)
                disease_to_alias.append([disease, alias])  # 存储“疾病”和“别名”的关系

            # 实体-发病部位(part):存在多个部位
            part_list = str(data[2]).strip().split() if str(data[2]) else "未知"
            for part in part_list:
                parts.append(part)
                diseases_to_part.append([disease, part])  # 存储“疾病”和“部位”的关系

            # 属性-发病人群(age):存在多个年龄段
            age = str(data[3]).strip()
            disease_dict["age"] = age

            # 属性-传染性(infection)
            infect = str(data[4]).strip()
            disease_dict["infection"] = infect

            # 属性-医保(insurance)
            insurance = str(data[5]).strip()
            disease_dict["insurance"] = insurance

            # 实体-所属科室(department):存在多个科室
            department_list = str(data[6]).strip().split()
            for department in department_list:
                departments.append(department)
                disease_to_department.append([disease, department])

            # 属性-检查项(checklist):存在多个检查项
            check = str(data[7]).strip()
            disease_dict["checklist"] = check

            # 实体-症状(symptom):存在多个症状,且每行最后有个“[详细]”
            symptom_list = str(data[8]).replace("...", " ").strip().split()[:-1]
            for symptom in symptom_list:
                symptoms.append(symptom)
                disease_to_symptom.append([disease, symptom])

            # 实体-并发症(complication):存在多个并发症
            complication_list = str(data[9]).strip().split()[:-1] if str(data[9]) else "未知"
            for complication in complication_list:
                complications.append(complication)
                disease_to_complication.append([disease, complication])

            # 属性-治疗方法(treatment)：存在多种治疗方法，最后还有个多余的字符串“[详细]”
            treat = str(data[10]).strip()[:-4]
            disease_dict["treatment"] = treat

            # 实体-药品(drug):存在多种药品
            drug_string = str(data[11]).replace("...", " ").strip()
            for drug in drug_string.split()[:-1]:
                drugs.append(drug)
                disease_to_drug.append([disease, drug])

            # 属性-治愈周期(period)
            period = str(data[12]).strip()
            disease_dict["period"] = period

            # 属性-治愈率(rate)
            rate = str(data[13]).strip()
            disease_dict["rate"] = rate

            # 属性-费用(money)
            money = str(data[14]).strip() if str(data[14]) else "未知"
            disease_dict["money"] = money

            diseases_infos.append(disease_dict)

        # 7种实体，6种关系，8种属性(字典组成的列表)
        return set(diseases), set(symptoms), set(aliases), set(parts), set(departments), set(complications), set(drugs), disease_to_alias, disease_to_symptom, diseases_to_part, disease_to_department, disease_to_complication, disease_to_drug, diseases_infos

    def create_node(self, label, nodes):
        """
        创建节点
        :param label: 标签
        :param nodes: 节点
        :return:
        """
        print("MedicalGraph.create_node()......")
        count = 0
        for node_name in nodes:
            node = Node(label, name=node_name)
            self.graph.create(node)
            count += 1
            print(count, len(nodes))

        return

    def create_diseases_nodes(self, disease_info):
        """
        创建疾病节点的属性
        :param disease_info: list(Dict)
        :return:
        """
        print("MedicalGraph.create_diseases_nodes()......")
        count = 0
        for disease_dict in disease_info:

            # py2neo.Node():为每一种疾病(行)创建一个节点
            # *labels: 定义节点类型；property:以key和value的形式定义节点属性
            node = Node("Disease", name=disease_dict['name'], age=disease_dict['age'],
                        infection=disease_dict['infection'], insurance=disease_dict['insurance'],
                        treatment=disease_dict['treatment'], checklist=disease_dict['checklist'],
                        period=disease_dict['period'], rate=disease_dict['rate'],
                        money=disease_dict['money'])
            self.graph.create(node)
            count += 1
            print(count, len(disease_info))

        return

    def create_graphNodes(self):
        """
        创建知识图谱实体
        :return:
        """
        print("MedicalGraph.create_graphNodes()......")
        disease, symptom, alias, part, department, complication, drug, rel_alias, rel_symptom, rel_part, rel_department, rel_complication, rel_drug, rel_infos = self.read_file()

        self.create_diseases_nodes(rel_infos)  # 创建“疾病”节点（同时添加属性）
        self.create_node("Symptom", symptom)  # 创建“症状”节点
        self.create_node("Alias", alias)  # 创建“别名”节点
        self.create_node("Part", part)  # 创建“发病部位”节点
        self.create_node("Department", department)  # 创建“所属科室”节点
        self.create_node("Complication", complication)  # 创建“并发症”节点
        self.create_node("Drug", drug)  # 创建“药品”节点

        return

    def create_graphRels(self):
        """
        创建知识图谱中实体之间的关系
        :return:
        """
        print("MedicalGraph.create_graphRels()......")

        disease, symptom, alias, part, department, complication, drug, rel_alias, rel_symptom, rel_part, rel_department, rel_complication, rel_drug, rel_infos = self.read_file()

        self.create_relationship("Disease", "Alias", rel_alias, "ALIAS_IS", "别名")
        self.create_relationship("Disease", "Symptom", rel_symptom, "HAS_SYMPTOM", "症状")
        self.create_relationship("Disease", "Part", rel_part, "PART_IS", "发病部位")
        self.create_relationship("Disease", "Department", rel_department, "DEPARTMENT_IS", "所属科室")
        self.create_relationship("Disease", "Complication", rel_complication, "HAS_COMPLICATION", "并发症")
        self.create_relationship("Disease", "Drug", rel_drug, "HAS_DRUG", "药品")

    def create_relationship(self, start_node, end_node, edges, rel_type, rel_name):
        """
        创建实体关系边
        :param start_node:关系起点
        :param end_node:关系终点
        :param edges:边数据
        :param rel_type:边类型
        :param rel_name:边名
        :return:
        """
        print("MedicalGraph.create_relationship()......")

        count = 0

        # 去重处理(利用了set()的特性：可变、无序、不重复)
        set_edges = []
        for edge in edges:
            set_edges.append('###'.join(edge))
        all = len(set(set_edges))
        for edge in set(set_edges):
            edge = edge.split('###')
            p = edge[0]  # 疾病(实体)
            q = edge[1]  # 别名(实体)
            query = "match(p:%s),(q:%s) where p.name='%s'and q.name='%s' create (p)-[rel:%s{name:'%s'}]->(q)" % (start_node, end_node, p, q, rel_type, rel_name)
            try:

                # py2neo.Graph().run(query):~py2neo中运行一个读/写查询
                self.graph.run(query)
                count += 1
                print(rel_type, count, all)
            except Exception as e:
                print(e)
        return


if __name__ == "__main__":
    handler = MedicalGraph()
    handler.create_graphNodes()
    handler.create_graphRels()
