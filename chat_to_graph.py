# -*- coding: utf-8 -*-
# @Author : maxin-nk
# @Email : xxin_ma@163.com
# @File : chat_to_graph

'''
生成大纲，通过关键词组、描述性内容与指定结构，生成对应的大纲
'''
from AI_tools.aiToolCalls import aiToolCalls
from kbqa_test import KBQA

memory = ''  # 记录用户的历史交互


class chat_to_graph:
    def __init__(self, aiType='sparkAI'):
        self.ai = aiToolCalls(aiType)

    def generatePrompt(self, query='', text_string='', qa_type="text", filePath=None):
        """
        根据结果生成大模型的prompt
        :param query:用户原始查询
        :param text_string: 知识图谱查询的结果
        :param qa_type: 输入内容的格式
        :param filePath: 输入的文件路径
        :return:
        """
        pr = ''
        if text_string == '' and qa_type == 'text':
            pr = ('作为一名用户的私人医疗健康管理助手，请人性化的回复用户的以下问题:{}。'
                  '该回答应当遵循以下原则：'
                  '第一，提供的回答应基于可靠的医学知识和最新的科学研究，避免提供未经证实的信息或误导性建议；'
                  '第二，提供积极的鼓励和支持，帮助用户感到被理解和关心。'
                  # '先根据以上两条原则给出回答，然后在回答的最后强调以下两点'
                  # '第一，建议用户在遇到健康问题时，特别是症状加重或出现新症状时，应及时就医寻求专业帮助；'
                  # '第二，告知用户模型的能力和局限性，无法替代专业的医疗建议。'
                  # '用户的咨询历史是：{}'
                  ).format(query, memory)
        elif text_string != '' and qa_type == 'text':
            pr = ('作为一名用户的私人医疗健康管理助手，请将以下知识（{}）整理为一段话，人性化的回复用户的以下问题：{}。'
                  '注意，解释应当遵循以下原则：1.解释应基于可靠的医学知识和最新的科学研究，避免提供未经证实的信息或误导性建议；'
                  '2.提供积极的鼓励和支持，帮助用户感到被理解和关心。'
                  # '先根据以上两条原则给出回答，然后在回答的最后强调以下两点：'
                  # '第一，建议用户在遇到健康问题时，特别是症状加重或出现新症状时，应及时就医寻求专业帮助；'
                  # '第二，告知用户模型的能力和局限性，无法替代专业的医疗建议。'
                  # '用户的咨询历史是：{}'
                  ).format(text_string, query, memory)
        elif qa_type == 'file':
            # pr = f'''现在假设你是一个医院质控管理人员，请查看以下文件:{filePath}，请依据以上文本，生成以上文本的 neo4j 格式知识图谱：'''
            pr = '作为一名用户的私人医疗健康管理助手，请查看以下文件，并生成此文件的 neo4j 格式知识图谱：{}'.format(filePath)

        return pr

    def knowledgeGraph(self, text_string=None, qa_type="text", top_nam = 10, filePath=None):
        """
        返回知识图谱的查询结果
        :param text_string: 用户的query文本
        :param qa_type:
        :param filePath:
        :return:
        """
        return KBQA(top_name=top_nam).qa_main(text_string)

    def generateGraph(self, text_string, top_name=10):
        """
        生成输出结果
        :param text_string:
        :param top_name:
        :return:
        """

        # 返回知识图谱查询结果
        res = self.knowledgeGraph(text_string, top_nam=top_name)
        print(res)

        # 生成提示词
        res_prompt = self.generatePrompt(query=text_string, text_string=res)

        # 调用aiType对应大模型，返回结果
        return self.ai.dispatchCenter(res_prompt)


if __name__ == '__main__':

    chat_to_graph = chat_to_graph(aiType='sparkAI')

    while True:
        inputs = input("用户：")

        if not inputs:
            break

        print("小鑫同学：")
        print(chat_to_graph.generateGraph(text_string=inputs, top_name=10))
        print("-" * 100)

        memory += inputs

    # # 常见问题测试：单意图
    # print(EntityExtractor().extractor('我最近总头晕是怎么回事？'))  # 0(query_disease)
    # print(EntityExtractor().extractor('地中海贫血有什么症状？'))  # 1(query_symptom)
    # print(EntityExtractor().extractor('地中海贫血怎么治疗？'))  # 2(query_cureway)
    # print(EntityExtractor().extractor('地中海贫血需要做哪些检查？'))  # 3(query_checklist)
    # print(EntityExtractor().extractor('地中海贫血挂什么科？'))  # 4(query_department)   -------------------------
    # print(EntityExtractor().extractor('地中海贫血的治愈率高吗？'))  # 5(query_rate)
    # print(EntityExtractor().extractor('地中海贫血多久能治好？'))  # 6(query_period)
    # print(EntityExtractor().extractor('详细介绍一下地中海贫血？'))  # 7(disease_describe)


    # 常见问题测试：多意图
    # print(EntityExtractor().extractor('地中海贫血有什么症状？多长时间能治好？'))  # 0(query_disease), 6(query_period)
    # print(EntityExtractor().extractor('地中海贫血的治疗方法有哪些?可以治吗?'))  # 2(query_cureway), 5(query_rate)

    # 极端问题测试
    # print(EntityExtractor().extractor('你是谁？'))
    # print(EntityExtractor().extractor('你真是个大傻叉！'))
    # print(EntityExtractor().extractor('转人工！'))
    # print(EntityExtractor().extractor('&……￥#*%￥！'))
    # print(EntityExtractor().extractor('昨天发烧，服用了阿司匹林,并且还吃了牛黄清胃丸，饭是吃了瓜烧白菜，大便有点色浅，可能是什么病？'))
