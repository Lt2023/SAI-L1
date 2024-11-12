from model import LanguageModel

def main():
    """
    主函数，用于与用户进行交互式对话。
    
    该函数创建了一个LanguageModel实例，并进入一个无限循环，直到用户输入'exit'。
    在循环中，它接收用户的输入，将用户的提问传递给语言模型，并打印模型的回答。
    """
    # 创建LanguageModel实例
    model = LanguageModel()  
    # 提示用户模型加载完成
    print("加载训练好的模型...")

    while True:
        # 提示用户输入问题，并提供退出程序的指令
        question = input("你好！请问你有什么问题？输入 'exit' 退出程序。\n")
        
        # 检查用户是否输入了退出指令
        if question.lower() == 'exit':  
            # 提示用户程序将要退出
            print("感谢使用！退出程序...")
            # 退出循环，终止程序
            break

        # 使用模型获取问题的唯一答案
        response = model.get_unique_answer(question)  
        # 打印模型的回答
        print(f"模型的回答：{response}\n")

# 确保main函数在程序直接运行时被调用
if __name__ == "__main__":
    main()
