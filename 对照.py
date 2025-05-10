# 示例3：教育领域问答系统（基于OpenAI API）
from openai import OpenAI
from dotenv import load_dotenv
import os

class EducationQASystem:
    def __init__(self, model_name="gpt-3.5-turbo"):
        load_dotenv()
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url="https://api.fe8.cn/v1"
        )
        self.model = model_name
    
    def generate_answer(self, question, subject="数学", max_length=512):
        # 构建领域特定提示
        # prompt = f"""
        # 你是一位专业的{subject}教师，擅长清晰、简洁地回答学生的问题。
        # 请回答以下问题：
        # {question}
        # 回答：
        # """
        prompt = question
        
        # 调用OpenAI API
        chat_completion = self.client.chat.completions.create(
            messages=[
                {"role": "user", "content": prompt}
            ],
            model=self.model,
            temperature=0.3,
            max_tokens=max_length
        )
        
        return chat_completion.choices[0].message.content.strip()

# 使用示例
if __name__ == "__main__":
    # 教育问答系统
    edu_qa = EducationQASystem()
    math_question = "如何求解二元一次方程组？"
    answer = edu_qa.generate_answer(math_question)
    print("教育问答结果:", answer)    


# sample output:
# 教育问答结果: 要求解二元一次方程组，首先需要将方程组中的两个方程进行整理，使得其中一个未知数的系数相等，然后通过加减消元法或代入法来求解。具体步骤如下：

# 1. 将方程组中的两个方程表示为标准形式，即将未知数和常数项移到等号的另一侧，使得方程等号右侧为0。例如，方程组为：
#    a1x + b1y = c1
#    a2x + b2y = c2

# 2. 通过乘法，使得其中一个未知数的系数相等。可以选择将第一个方程乘以b2，第二个方程乘以b1，得到：
#    b2(a1x + b1y) = b2c1
#    b1(a2x + b2y) = b1c2

# 3. 将上述两个方程相减，消去y，得到一个只含有x的方程：
#    b2a1x - b1a2x = b2c1 - b1c2
#    (b2a1 - b1a2)x = b2c1 - b1c2

# 4. 求解上述方程得到x的值。

# 5. 将求得的x的值代入任意一个原方程中，求解得到y的值。

# 6. 最终得到方程组的解为(x, y)。

# 希望以上步骤能帮助你更好地理解如何求解二元一次方程组。如果有任何疑问，请随时向我提问。