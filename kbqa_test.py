# -*- coding: utf-8 -*-
# @Author : maxin-nk
# @Email : xxin_ma@163.com
# @File : kbqa_test(启动问答测试)

from entity_extractor import EntityExtractor
from search_answer import AnswerSearching
import numpy as np


class KBQA:
    def __init__(self, top_name=10):
        self.extractor = EntityExtractor()
        self.searcher = AnswerSearching(top_name)

    def qa_main(self, query):
        answer = ['对不起，您的问题我不知道，我今后会努力改进的!', '这个问题让我感觉像是猫看到了彩虹，完全搞不清楚！不过我可以帮你查找资料，或者换个问题让我试试？',
                  '看来你问了一个能把我难倒的问题！或许这是个Bug，不过我很乐意帮你找找答案!', '这个问题就像是在问一只金鱼如何驾驶汽车！我可能无法回答，但我可以帮你找找看!',
                  '这个问题简直比解一道高数题还难！不过我可以帮你搜集相关信息，看看能不能找到答案!', '你这个问题有点像是要我解释宇宙的终极奥秘！我还没那么聪明，不过我可以帮你查找相关信息!',
                  '这个问题对我来说就像是迷宫中的迷宫，我还在找出口呢！但我可以帮你找找相关资料。', '哎呀，这个问题超出了我的智商上限，看来我得再去上几节课才能回答你了！',
                  '嗯，这个问题对我来说就像是外星语言一样神秘！要不你再给我点提示？', '我的电路有点短路了，对这个问题的答案一时半会儿找不到。要不我们一起探索一下？',
                  '看来这个问题难度系数有点高，甚至我的电源都冒烟了！', '这个问题真是个烧脑的问题啊，我得让我的处理器冷静一下再回答你。你愿意稍等一会儿吗？',
                  '哇，这个问题让我感到有点晕头转向，可能需要升级一下我的软件才能回答你！', '哇塞，这个问题就像是黑洞，吸走了我所有的智慧！要不你再给我点线索？', '看来我的知识库还需要补课，这个问题我暂时还没答案。不过，我们可以一起寻找答案！',
                  '嗯，这个问题就像是给了一台微波炉一个计算质能方程的任务，我有点吃不消呢！要不我们换个话题？']

        # 抽取用户query中的实体
        entities = self.extractor.extractor(query)
        if not entities:
            return answer[np.random.randint(0, len(answer))]

        # 根据不同的实体和意图构造cypher查询语句
        sqls = self.searcher.question_parser(entities)

        # 执行cypher查询，返回结果
        final_answer = self.searcher.searching(sqls)
        if not final_answer:
            # return answer[np.random.randint(0, len(answer))]
            return ''
        else:
            return '\n'.join(final_answer)


if __name__ == "__main__":
    handler = KBQA(top_name=10)
    while True:
        question = input("用户：")
        if not question:
            break
        answer = handler.qa_main(question)
        print("小鑫同学：", handler.extractor.extractor(question))
        # print("小鑫同学：", "(温馨提醒：结果仅供参考！)".join(answer))
        print("小鑫同学：", answer)
        print("*"*50)

    # # 常见问题测试：单意图
    # print(EntityExtractor().extractor('我最近总头晕是怎么回事？'))  # 0(query_disease)
    # print(EntityExtractor().extractor('地中海贫血的治愈率高吗？'))  # 5(query_rate)
    # print(EntityExtractor().extractor('地中海贫血有什么症状？'))  # 1(query_symptom)
    # print(EntityExtractor().extractor('地中海贫血怎么治疗？'))  # 2(query_cureway)
    # print(EntityExtractor().extractor('地中海贫血挂什么科？'))  # 4(query_department)   -------------------------
    # print(EntityExtractor().extractor('地中海贫血多久能治好？'))  # 6(query_period)
    # print(EntityExtractor().extractor('详细介绍一下地中海贫血？'))  # 7(disease_describe)
    # print(EntityExtractor().extractor('地中海贫血需要做哪些检查？'))  # 3(query_checklist)

    # 常见问题测试：多意图
    # print(EntityExtractor().extractor('什么是地中海贫血？多长时间能治好？'))  # 0(query_disease), 6(query_period)
    # print(EntityExtractor().extractor('地中海贫血的治疗方法有哪些?可以治吗?'))  # 2(query_cureway), 5(query_rate)

    # 极端问题测试
    # print(EntityExtractor().extractor('你是谁？'))
    # print(EntityExtractor().extractor('你真是个大傻叉！'))
    # print(EntityExtractor().extractor('转人工！'))
    # print(EntityExtractor().extractor('&……￥#*%￥！'))
    # print(EntityExtractor().extractor('昨天发烧，服用了阿司匹林,并且还吃了牛黄清胃丸，饭是吃了瓜烧白菜，大便有点色浅，可能是什么病？'))
