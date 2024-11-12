from model import LanguageModel

def main():
    """
    主函数，负责实例化语言模型并进行训练
    """
    # 实例化语言模型
    model = LanguageModel() 
    # 提示开始训练模型
    print("开始训绨模型...")
    # 训练模型，设置训练轮数为1000轮
    model.train(epochs=1000)  
    # 训练完成后，打印提示信息
    print("训练完成！")

# 当作为主模块执行时，调用main函数
if __name__ == "__main__":
    main()