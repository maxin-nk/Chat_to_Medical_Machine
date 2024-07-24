# -*- coding: utf-8 -*-
# @Time : 2024/1/31 15:46 
# @Author : maxin-nk
# @Email : xxin_ma@163.com


'''
项目：访问openai，方便进行快速调试使用
'''
from openai import OpenAI
import initialization


class chat_openai:

    OPENAI_API_KEY = initialization.openAI_key
    # base_url = initialization.openAI_base_url
    modelType = "gpt-3.5-turbo"

    def __init__(self):
        self.client = OpenAI(
            api_key=self.OPENAI_API_KEY,  # this is also the default, it can be omitted
            # base_url=self.base_url
        )
        # self.prompt = """
        # 你好，请问你是？
        # """

    def chat(self, text):
        resp = self.client.chat.completions.create(
            model=self.modelType,
            messages=[{"role": "user", "content": text}],
        )
        return resp


if __name__ == '__main__':

    # 测试
    inputs = input("用户：")
    req = chat_openai().chat(inputs)
    # print(req)
    # print(req.json)
    print(req.choices[0].message.content)
