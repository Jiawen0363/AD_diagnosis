from data_utils import Preprocessor
from transformers import AutoTokenizer
import json

# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained('/code/models/Llama-3.2-3B-Instruct')

# 创建预处理器
preprocessor = Preprocessor(
    tokenizer=tokenizer,
    max_length=2048,
    template_name='alpaca',
    mask_dtype='float32'
)

# 测试多轮对话数据
multi_turn_data = {
    'instruction': 'You are a college tutor specializing in Python programming.',
    'conversations': [
        {'from': 'human', 'value': '你好，我想学习Python'},
        {'from': 'gpt', 'value': '你好！很高兴帮助你学习Python...'},
        {'from': 'human', 'value': '我想了解类和对象的概'''''},
...'},
        {'from': 'human', 'value': '能给我一个具体的例子吗？'},
        {'from': 'gpt', 'value': '当然！这里有一个简单的例子...'}
    ]
}

try:
    result = preprocessor.preprocess(multi_turn_data)
    print('✅ 多轮对话处理成功!')
    print('处理结果:', result.keys())
    
    # 检查输入和标签
    input_ids = result['input_ids']
    labels = result['labels']
    
    print(f'输入长度: {len(input_ids)}')
    print(f'标签长度: {len(labels)}')
    
    # 解码看看内容
    input_text = tokenizer.decode(input_ids)
    print('\n输入内容:')
    print(input_text[:500] + '...')
    
    # 检查哪些部分被mask了
    ignore_token_id = -100
    masked_positions = (labels == ignore_token_id).sum().item()
    total_positions = len(labels)
    print(f'\nMasked positions: {masked_positions}/{total_positions}')
    
    # 检查标签中非mask的部分
    non_masked_labels = labels[labels != ignore_token_id]
    if len(non_masked_labels) > 0:
        target_text = tokenizer.decode(non_masked_labels)
        print('\n训练目标 (非mask部分):')
        print(target_text[:200] + '...')
    
except Exception as e:
    print('❌ 多轮对话处理失败')
    print('错误:', str(e))
