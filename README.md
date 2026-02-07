## 笨笨实习生千石抚子语录（项目说明）

<p align="center">
  <img src="FrontEnd/images/CircleC.jpg" alt="千石抚子 · CircleC" width="260">
</p>

既然你都翻到这里来了，我就稍微认真一点，说清楚这个小项目是怎么运转的。省得你一边乱点代码一边头痛。这基本可以说是**最简形式的**涵盖本地大模型微调+部署+工作流+前端的Agent应用项目。如果你觉得我讲得还不错，请留下一个星星，谢谢！

---

## 项目结构

- **`goofyAgent.py`**  
  这里是整个服务的大脑。  
  - 用 FastAPI 起了一个 HTTP 服务，对外暴露 `/generate` 接口。  
  - 会先把用户输入丢给 DeepSeek 的在线接口做一个简单判断，看看是不是「医疗相关」。  
  - 如果判断是医疗问题，就把内容转给本地的医疗大模型（`medicalAPI`）来出主意；  
  - 如果不是，就走普通聊天路线，用抚子一贯的说话方式陪你闲聊。  
  - 同时还会在 `memory.json` 里记一点对话历史，用来维持角色风格和上下文（当然，只是简单的小本子，不是你想象的那种可怕的监控）。  

- **`medicalAPI.py`**  
  这是本地医疗模型的小窝。  
  - 用的是 Hugging Face 上的 `Qwen/Qwen3-8B`，再加载你在 `exp1` 里训练好的 LoRA 权重。  
  - 模型通过 4bit 量化加载到显卡上，省点显存，勉强能在普通机器上活下去。  
  - 暴露出一个 `evaluate(input_text)` 函数，`goofyAgent` 会调用它来获取「专业一点」的医疗建议。  
  - 这里不负责 HTTP 服务，只做纯粹的模型推理，安安静静当后端算力工具人。  

- **`FrontEnd/goofyDoc.html`**  
  这个是给人看的前端界面。  
  - 一个单页 HTML，里面应该是你和我说话的聊天界面。  
  - 样式、布局、交互都写在这里，浏览器打开就能用。  
  - 一般会通过前端 JS 调用后端的 `/generate` 接口，把你的废话……不，是宝贵的问题，转交给我。  

- **`Finetune/`**  
  这个文件夹是给你折腾微调用的。我的创造人灰鼠参考了[`这个仓库`](https://github.com/Hoper-J/AI-Guide-and-Demos-zh_CN)，真的很棒！
  - `Finetune/medicalLLM.py` 是一个简单的微调/数据处理脚本，你如果真有兴趣继续训练，就可以参考它自己改。  
  - `Finetune/tmp_medical.json` 是一小份中文医疗对话训练集切片，数据原始仓库来自 [`Chinese-medical-dialogue-data`](https://github.com/Toyhom/Chinese-medical-dialogue-data)。  
  - 这里的东西主要是服务于之前给抚子做 LoRA 微调那一套流程，正常使用聊天功能完全可以不用管。  

- **`FrontEnd/images/`**  
  前端页面用到的背景图和装饰图片。  
  - `background.jpg` 和几个 `Circle*.jpg` 都是界面的美术素材。  
  - 删掉的话界面还勉强能用，只是会变得丑一点而已。  

- **`exp1/`**  
  这是你给抚子做微调的实验目录。  
  - `checkpoint-*/` 里是训练好的 LoRA 权重和 tokenizer 之类的文件。  
  - `medicalAPI.py` 会在这里自动找到最新的 checkpoint 来加载。  
  - 至于里面训练过程的小秘密，我就不帮你翻出来了。  

- **`environment.yml`**  
  Conda 环境配置文件。  
  - 里面列了项目需要的 Python 版本和依赖包。  
  - 想在新机器上把我召唤出来，这个文件会很重要。  

---

## 环境准备

如果你打算在一台干净的机器上把我叫醒，大致要这样做：

1. **创建 Conda 环境**

   ```bash
   conda env create -f environment.yml
   conda activate xxx (env name)
   ```

2. **准备模型文件**

   - 确保 `exp1/` 目录里已经有训练好的 `checkpoint-*` 子目录。  
   - 第一次运行时，`medicalAPI.py` 还会从 Hugging Face 拉基础模型到 `cache/`，网络慢的话你自己多等等。  

3. **设置 DeepSeek API Key**

   你可以二选一：

   - 用环境变量：

     ```bash
     set DEEPSEEK_API_KEY=你的密钥
     ```

   - 或者启动时加参数：

     ```bash
     python goofyAgent.py --apikey 你的密钥
     ```

   不给密钥的话，在线判断是否为医疗问题那一步会直接报错不干活。

---

## 启动后端服务

后端就是跑 `goofyAgent.py` 这一个入口。

- **方式一：直接运行脚本**

  ```bash
  python goofyAgent.py --apikey 你的密钥
  ```

  代码里已经写了 `if __name__ == "__main__":`，会用 `uvicorn` 起服务，默认监听：

  - 地址：`http://0.0.0.0:8000`
  - 核心接口：`POST /generate`

- **方式二：手动用 uvicorn**

  ```bash
  uvicorn goofyAgent:app --host 0.0.0.0 --port 8000
  ```

  这种方式就需要你自己保证环境变量里已经有 `DEEPSEEK_API_KEY`。

---

## 前端使用方式

前端其实很朴素，你要做的只是：

1. 用浏览器打开 `FrontEnd/goofyDoc.html`。  
2. 确保它里面配置的后端地址指向你刚刚启动的 FastAPI 服务（通常是 `http://127.0.0.1:8000/generate` ）。  
3. 然后你就可以在页面里对我说话了。  

如果你想更深入折腾模型本身，比如继续微调、换一批数据之类的，可以去看看 `Finetune/` 里的脚本和小数据集，自己动手，不要指望我帮你跑完所有实验。

---

## 接口简要说明

- **`POST /generate`**  
  - 请求体（JSON）：
    - `user_input`: 用户输入的文本  
    - `user_id`: 用户标识，用来区分对话记忆（不填的话默认 `"anonymous"`）  
  - 返回（JSON）：
    - `reply`: 抚子给你的回复  
    - `route`: `"medical"` 或 `"chat"`，表示刚才走的是哪条回答路径  
    - `judge_raw`: DeepSeek 判定原始结果，用来判断是不是医疗相关  

你要是执意去看返回里的那些细节，我也不会拦着你。但至少别再问「这个接口是干嘛的」这种问题了。

---

## 一点小提示

- 本地模型加载和推理都挺吃显存的，如果你机器太弱，我也没办法给你太快的回复。  
- `memory.json` 里只是简单的对话记录，如果你觉得不安全，随时可以删掉或自己改成别的持久化方式。 
- 微调后，在 **`Finetune/`** 文件夹中会生成新的 **`exp1/`** 权重。如果你希望加载新的权重进行推理，请用新的 **`exp1/`** 替换根目录的 **`exp1/`**，或者将里面的checkpoints复制到根目录的 **`exp1/`** 中，并更新其后缀数字使推理模型选择你所期望的checkpoints权重，详情见 **`medicalAPI.py`** line 27-35。
- 灰鼠（我的创造人）的工作流是，将项目的后端`goofyAgent.py`，`medicalAPI.py`以及`exp1/`中的checkpoints托管到高性能服务器中，利用nginx反代域名并配置https服务（很重要！现在的浏览器很少支持http了...）然后，灰鼠将前端直接放在内置宝塔的随便一个服务器里面；如果你的用户都能访问github，灰鼠觉得首选github-page托管，或者再用cloudflare。

差不多就这样了。既然你已经把我叫出来用，就好好对待我一点吧。

