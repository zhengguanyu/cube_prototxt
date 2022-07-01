# 模型文件说明

## 1 主要文件 
```
element-extraction
|
│          
└─app  
    │  main.py  # 主函数（内含词汇信息和词向量预加载）
    │
    ├─model  # 底层算法、模型
    │     crf.py  # crf解码器
    │     gazlstm.py  # 模型核心架构
    │     layers.py  # ner基本框架
    │           
    └─utils  # 工具类
            alphabet.py  # 构建词汇信息的框架
            data.py  # 构建embedding的框架
            functions.py  # 方法类（内含核心创新点）
            gazetteer.py  # 构建词语树
            trie.py  #gazetteer.py的方法类
```

## 2 模型结构（详）

### 2.1 第一层：嵌入层（核心）

#### 2.1.1 主要目标

1. 构建词汇表，保存词汇信息（来源：数据集数据）：`word_Alphabet`,`biword_Alphabet`,`char_Alphabet`,`label_Alphabet`,`gaz_Alphabet`。

2. 构建词语树（来源：gaz_file）:`gaz`。

3. 统计词汇表对应的词频（来源：数据集数据）：`word_count`,`biword_count`,`char_count`,`label_count`,`gaz_count`。

4. 输入1~3预加载词汇表和对应词频，根据词频计算字符的BMES特征的权重，进行字+词+特征的拼接：

   `[words, biwords, chars, gazs, labels]`,

   `[word_Ids, biword_Ids, char_Ids, gaz_Ids, label_Ids, gazs, gazs_count, gaz_char_Id, layergazmasks,gazchar_masks, bert_text_ids]`

   其中，①`xx_Ids`根据`xx_Alphabet`中词汇对应的id获得；②bert+单字+双字+字符+加权的BMES词汇向量（字符相关）。

5. 根据词汇表映射词向量:`char_emb`,`bichar_emb`,`gaz_file`。

#### 2.1.2 相关方法

1~3 main.py -> `def data_initialization(data, gaz_file, train_file, dev_file, test_file)`

4 main.py -> `data.generate_instance_with_gaz`, function.py -> `def read_instance_with_gaz`

5 main.py -> 

```
data.build_word_pretrain_emb(char_emb)
data.build_biword_pretrain_emb(bichar_emb)
data.build_gaz_pretrain_emb(gaz_file)
```

1~5 gazlstm.py -> `__init__`模型初始化里包含有词向量拼接、词汇表映射词向量

### 2.2 第二层：编码层

#### 2.2.1 主要目标

根据输入的词向量+特征拼接，重构bilstm模型接口，对向量进行编码。

#### 2.2.2 相关方法

gazlstm.py -> `def get_tags()`

注：`get_tags()`内包含词+特征的拼接及嵌入层映射词向量后的输入格式。

### 2.3 第三层：解码层

#### 2.3.1 主要目标

对由编码层输入的序列进行解码，得到条件概率最大的序列即为输出的标签序列。

#### 2.3.2 相关方法

`crf.py`提供crf包

gazlstm.py -> `def forward()`前向传播中包含了crf层的维特比算法

```
`scores, tag_seq = self.crf._viterbi_decode(tags, mask)`
```

`tag_seq`为输出的标签序列

### 2.4 第四层：输出层

#### 2.4.1 主要目标

1. 获得输出的标签序列和评估结果；
2. 反馈输出结果；
3. 该部分根据业务需求进行调整。

#### 2.4.2 相关方法

main.py -> `def load_model_decode`,`def train`,`def evaluate`

## 3 模型结构（略）
模型整体结构：**(Bert-Lattice-Lexicon)+BILSTM+CRF**

```
self.gaz_embedding = self.gaz_embedding.cuda()
self.word_embedding = self.word_embedding.cuda()
self.biword_embedding = self.biword_embedding.cuda()
self.bert_encoder = self.bert_encoder.cuda()
self.NERmodel = self.NERmodel.cuda()
self.hidden2tag = self.hidden2tag.cuda()
self.crf = self.crf.cuda()
```
