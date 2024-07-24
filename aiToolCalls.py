# -*- coding: utf-8 -*-
# @Author : maxin-nk
# @Email : xxin_ma@163.com
# @File : aiToolCalls
'''
调用ai工具，负责对接ai工具的调用（如openai、文心一言等对接方切换，就在这里进行）
使用：传入文本，返回处理后的文本
'''
from AI_tools.openAI_chat import chat_openai
from AI_tools.spartAI_chat import chat_spark
import AI_tools.spartAI_chat as spartAI_chat


class aiToolCalls:
    def __init__(self, aiType='sparkAI'):
        self.aiType = aiType

    def openAI(self, text):
        """
        取出返回内容的内容结论部分
        :param text: 用户的query文本
        :return:
        """
        return chat_openai().chat(text).choices[0].message.content

    def sparkAI(self, text):
        chat_spark(text)
        res = spartAI_chat.answer
        spartAI_chat.answer = ''
        return res

    def GLM(self, text):
        pass

    def baidu(self, text):
        pass

    def Claude(self, text):
        pass

    # 调度中心
    def dispatchCenter(self, text):
        """
        根据aiType类型，选择调用哪个大模型接口
        :param text:
        :return:
        """
        if self.aiType == 'openai':
            return self.openAI(text)
        elif self.aiType == 'sparkAI':
            return self.sparkAI(text)
        elif self.aiType == 'glm':
            return self.GLM(text)
        elif self.aiType == 'baidu':
            return self.baidu(text)
        elif self.aiType == 'claude':
            return self.Claude(text)
