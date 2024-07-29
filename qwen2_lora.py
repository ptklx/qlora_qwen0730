import warnings
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0" # 使用第一个GPU
os.environ['TRANSFORMERS_OFFLINE']='1' # 离线的方式加载模型
os.environ['DWAN _DISABLED'] = 'true' #模型规模较小，，单GPU，不需要zeRO优化来减少内存占用
warnings.filterwarnings("ignore")   #忽略所有警告
#ipynb 最方便
###  -i https://pypi.tuna.tsinghua.edu.cn/simple
# pip install modelscope   bitsandbytes
# pip install transformers datasets evaluate peft accelerate gradio optimum sentencepiece  
import torch
from modelscope import (AutoModelForCausalLM, BitsAndBytesConfig)

# python -m 

#BitsAndBytesconfig;配置模型最化参数的头
#4位量化(从32位浮点数转换为4位整数)，对比量化前后参数量的变化
_bnb_config = BitsAndBytesConfig(load_in_4bit=True, # 权重被加载为4位整数
                                bnb_4bit_use_double_quant=True,#双量化方案:对权重和激活值进行量化
                                bnb_4bit_quant_type="nf4",
                                # 量化(允许权重和激活值以不同的精度进行量化,
                                bnb_4bit_compute_dtype=torch.float32)# 量化后的int4的计算仍然在32位浮点精度(torch.float32)上进行,
#从预训练模型库中加载"0wen/owen2-0.5B"模型
_model = AutoModelForCausalLM.from_pretrained("Qwen1.5-0.5B-Chat",
                                                # 少cpu内存的使用
                                                low_cpu_mem_usage=True,#将量化配置_bnb config应用于模型加载
                                                quantization_config= _bnb_config)
#计算模型的总参数量# 
print(f"{sum(p.numel()for p in _model.parameters())}")

##################################
from modelscope import AutoTokenizer, AutoModel
#自动识别并加载与"Qwen/Qwen2-0.5B"模型匹配的的分词器
# #将文本转换为模型可以理解的数字序列(即ids)
# 词汇表:ids(索引)--tokens(词元
_tokenizer = AutoTokenizer.from_pretrained("Qwen1.5-0.5B-Chat")
##分词器将文本分割成词元，将词元转换为模型可以理解的IDS
ids = _tokenizer.encode("你是谁?爱谁谁!",return_tensors="pt")
tokens =_tokenizer.convert_ids_to_tokens(ids[0])
for id, token in zip(ids[0],tokens):
    print(f"{id}--{token}")
    #词向量#
#_model = AutoModel.from_pretrained("Qwen1.5-0.5B-Chat")
# 将ids转换为词向量# 
# embeddings =_model(ids)
# print(embeddings)

##% #################################
from datasets import load_dataset
#加载ison格式的训练数据集
_dataset = load_dataset("json", data_files="data.json", split="train")
#预处理数据集函数
def preprocess_dataset(example):
    MAX_LENGTH = 256 # 最大长度
    _input_ids,_attention_mask,_labels =[],[],[]# 初始化输入 ids、注意力掩码、标签列表
    #使用分词器对指令和响应进行编码
    _instruction = _tokenizer(f"user: {example['instruction']}Assistant:", add_special_tokens=False)
    print(_instruction)
    _response =_tokenizer(example["output"]+ _tokenizer.eos_token, add_special_tokens=False)
    #拼接指令和响应的输入ids
    _input_ids = _instruction["input_ids"]+ _response["input_ids"]
    #拼接指令和响应的注意力掩码
    _attention_mask = _instruction["attention_mask"]+ _response["attention_mask"]
    #拼接标签，这里将第一个指令的标签设置为-100
    _labels =[-100]* len(_instruction["input_ids"])+ _response["input_ids"]
    #如果拼接后的输入ids长度超过最大长度，则进行截
    if len(_input_ids)> MAX_LENGTH:
        _input_ids = _input_ids[:MAX_LENGTH]
        attention_mask = _attention_mask[:MAX_LENGTH]
        _labels =_labels[:MAX_LENGTH]
    return {
        "input_ids": _input_ids,
        "attention_mask":attention_mask,
        "labels":_labels
    }


#移除原始数据集中的列，只保留预处理后的数据。
_dataset = _dataset.map(preprocess_dataset, remove_columns=_dataset.column_names)
_dataset = _dataset.shuffle()


###############
from peft import (LoraConfig,get_peft_model,TaskType)
#定义微调配置
config = LoraConfig(task_type=TaskType.CAUSAL_LM, 
                    # 因果语言模型(模型的输出只依赖于当前的输入，而不依赖于未来的输入)
                    r=8,#LORA缩放因子，控制模型的稀疏性
                    target_modules="all-linear") # 所有权重都参与训练
#自动识别并加载与"owen/owen2-0.5B"模型匹配的微调配置
_model = get_peft_model(_model, config)

##########################
from transformers import TrainingArguments,Trainer,DataCollatorForSeq2Seq

#定义训练参数
_training_args = TrainingArguments(output_dir="checkpoints/qlora",
                                    # 训练结果的存储目录
                                    run_name="qlora_study",#运行的名称
                                    per_device_train_batch_size=10, # batch_size批处理大小
                                    num_train_epochs=6, #训练的轮次
                                    save_steps=6, # 保存检查点的轮次步数
                                    logging_steps=6, #写日志的轮次步数
                                    report_to="none",#指定不报告任何日志
                                    optim="paged_adamw_32bit") # 指定优化器
# 创建 Trainer 对象
trainer = Trainer(model=_model,
                  #指定模型对象
                  args=_training_args, #指定训练参数
                  train_dataset=_dataset,#指定训练数据集
                  data_collator=DataCollatorForSeq2Seq(tokenizer=_tokenizer,padding=True),
                    ) # 指定数据集的收集器

#调用 Trainer 的 train 方法开始训练
trainer.train()

################################
#将微调后的模型合并回原始模型，并最终保存更新后的模型
from transformers import pipeline ,AutoModelForSeq2SeqLM
from peft import PeftModel
#加载模型#使用
#AutoModelForcausalLm.from_pretrained方法自动识别并加载与"Qwen/Qwen2-8.5B"模型匹配的模型
_model = AutoModelForCausalLM.from_pretrained("Qwen1.5-0.5B-Chat",
                                              quantization_config=_bnb_config,# 指定量化配置
                                              low_cpu_mem_usage=True)#指定减少CPU内存使用
#使用PeftModel.from pretrained方法自动识别并加载与 model匹配的微调配置
# # model参数指定原始模型对象，model id参数指定微调配置的ID
peft_model = PeftModel.from_pretrained(model= _model, model_id="checkpoints/glora/checkpoint-100")
#使用pipeline方法创建一个管道对象，用于生成文本#pipeline方法需要三个参数:任务类型、模型对象、分词器对象
pipe = pipeline("text-generation", model=peft_model, tokenizer= _tokenizer)

#使用pipeline管道生成文本
# pipe(f"Ut:")ser: 你是谁? Assistan
pipe(f"Ut 你是谁 ？")

###################合并后的模型以供后续使用##
from transformers import pipeline
from peft import PeftModel
#加载模型
_model=AutoModelForCausalLM.from_pretrained("Qwen1.5-0.5B-Chat")
#model 参数指定原始模型对象，model_id 参数设置微调配置的ID
peft_model=PeftModel.from_pretrained(model=_model,
                                     model_id="checkpoints/qlora/checkpoint-6")


######################


