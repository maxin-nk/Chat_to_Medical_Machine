# Chat_to_Medical_Machine
基于深度知识图谱和预训练大语言模型的医疗问答系统

1.pretrained_bert_chinese文件夹下载地址：https://pan.quark.cn/s/7418340868fb

2.data/merge_sgns_bigram_char300.txt下载地址：https://pan.quark.cn/s/f0d0e94476f6

3.neo4j数据库的安装与配置：https://maxin.blog.csdn.net/article/details/140507035

4.运行：第一，运行build_graph.py,构建neo4j知识图谱（7类命名实体约3.7万，6种实体间关系约21万）；第二，运行和测试entity_extractor.py，提取用户query实体；第三，运行和测试search_answer.py，匹配和返回知识图谱结果；第四，运行和测试kbqa_test.py，同时运行第二、第三两步；第五，运行和测试chat_to_graph.py，引入大语言模型（目前仅支持ChaGPT和讯飞星火的Spark_Lite）

5.代码复用请标明出处

6.本项目参考：【https://github.com/zhihao-chen/QASystemOnMedicalGraph.git】【https://github.com/ligenxun/chatToMedicalAtlas.git】【https://github.com/Xiaoheizi2023/NLP_KBQA.git】

7.项目运行效果
7.1 知识图谱
![image](https://github.com/user-attachments/assets/5a0b9b18-ac7d-42bc-9065-754bf51b3d21)
7.2 问答系统输出
![image](https://github.com/user-attachments/assets/0606b4d9-773c-4211-b027-1d192ad59e9c)
