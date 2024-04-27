# OpenDispatcher and why model routing is really hard

Thank you very much to [@sershokunin](https://twitter.com/sershokunin?ref_src=twsrc%5Egoogle%7Ctwcamp%5Eserp%7Ctwgr%5Eauthor) for generously sponsoring this project!

Let's get right into it. Model routing is all about selecting the best LLM to serve a user's task. By best, I mean "cheapest that still gets the job done". For instance, GPT-4 costs roughly 20x more than GPT-3.5 per token so you want to avoid using GPT-4 on simple tasks, like "write me a bedtime story".

I have spent way too much $ on OpenAI credits so I wanted to figure out a way to intelligently route requests between LLMs to minimize cost while maintaining task completion rates. There are a few companies that are doing this as a service, such as [Martian](https://withmartian.com/). By the way, I ran into the people behind Martian at an event in SF and they weren't able to say too much but they did say that they had figured out a technique to express transformers as programs. More on this at the end, if you're interested.

Anyways, when I started working on my router, I knew that I wanted it to be as lightweight as possible, adding minimal time to inference. So, I initially wanted to avoid using any larger model, cough cough, Transformers.

But let's take a step back. I needed to get data first, specifically examples of prompts where some LLMs failed and some succeeded. Theoretically, the model router could learn to predict which LLMs would be able to succeed on a given prompt/task.

## Data
I had the idea to mine the [OpenLLM](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard) leaderboard on HuggingFace for free data. Basically, every time the evals are run on a new model, the per-question results are saved. So, I know that Mistral-7B failed Question 17 in GSM8K, for example. The OpenLLM leaderboard can only run in open source models so I was limited to results for models with weights available on Hugging Face, such as Mistral.

I decided that the first iteration of the model router would attempt to route between three open source models of increasing capability: Phi-2, Mistral-7B-Instruct-v0.2, and Mixtral-8x7B-Instruct-v0.1.
![Benchmark Scores by Model](https://cloud-12iix5eql-hack-club-bot.vercel.app/0benchmarkscoresbymodel.png)
Phi-2 is generally dumber than Mistral-7b, which in turn is generally dumber than Mixtral. I plotted their scores on six popular benchmarks to give you an idea. However, keep in mind that benchmarks are extremely unreliable due to "cheating" (training on benchmark data unintentionally). Still, I feel confident in saying that Phi-2 < Mistral-7b < Mixtral.

I downloaded the benchmark results for these three models and then  looked for prompts/questions that matched the hierarchy (Phi-2 < Mistral-7b < Mixtral). I ignored prompts that broke this hierarchy (such as a prompt that Phi-2 got right but Mixtral got wrong). I am interested in revisiting the soundness of ignoring hierarchy-breaking prompts.
![Questions that obeyed the performance hierarchy](https://cloud-dld3vdjg9-hack-club-bot.vercel.app/0screenshot_2024-04-26_at_9.44.22___pm.png)
I ended up with around ~25,000 prompts that I bucketed into four categories: easy, medium, hard, very hard. The easy category was for prompts that were answered correctly by all three models. The very hard category was for prompts that none of the models could answer.

## Training
Now, I had a very imbalanced dataset of prompt labeled by difficulty for LLMs. I wanted a machine learning model that could predict difficulty given a prompt. My first idea was to try multinomial regression using TF-IDF vectorization. This worked very poorly (classification report below). Random Forest didn't do any better. You can train the models yourself with this [Colab notebook](https://colab.research.google.com/drive/1OASjZB6qDjxLfC2iC_m4FY6IBAwcucSG?usp=sharing).

55% accuracy seems good for four-class classification (expected 25%), but it's only this high because of data imbalance. The Logistic Regression model was often rewarded for selecting "easy" since "easy" was the most common class. As you can see, the model does OK on "easy" but is horrendously bad on "hard" and "medium". Recall of 0.03 and 0.02!
![Logistic Regression Results](https://cloud-ekyto84fd-hack-club-bot.vercel.app/0screenshot_2024-04-26_at_10.10.07___pm.png)
Next, I tried using [textstat](https://pypi.org/project/textstat/) to calculate readability scores for each prompt and use those as features. This did not help.

At this point, I thought I was being too ambitious with four-class classification and tried just predicting between easy and everything else (questions Phi-2 can answer, and questions it can't). 

Even with this much easier task (and more data per class), I was not able to get good results. The classes were close to balanced and the overall accuracy was 61%. This is too bad to be used in practice.
![enter image description here](https://cloud-q6n1tlj5w-hack-club-bot.vercel.app/0screenshot_2024-04-26_at_10.16.36___pm.png)

## Pivot!
At this point, I gave up on not using Transformers and decided to try using a small version of BERT, distilBERT. It's much larger than the Logistic Regression and Random Forest models, but still small enough that you could do inference quickly.

Also, I had stumbled across a dataset called [RouterBench](https://huggingface.co/datasets/withmartian/routerbench), provided by the Martian team. It has pretty much the same structure as the dataset I assembled, except with the addition of benchmark results for closed source models like GPT-4 and Claude. I decided to pivot to training a model router that could select between GPT-4 and GPT-3.5. This seemed like it would be useful to more people, than Phi-2 vs. not Phi-2, since a lot of companies/API users are just using OpenAI's hosted models for convenience and quality. I stripped out non-English prompts because I didn't want to add more noise than necessary.

Finally, I decided to switch to multi-label classification instead of binary classification. Instead of having distilBERT output a single prediction (GPT-3.5 OR GPT-4), I had it output two values, essentially probability scores for GPT-3.5 and GPT-4 being able to answer the prompt. I thought this might be more intuitive for the model and it would also enable me to tune the thresholds later.

Specifically, I fine-tuned distilbert-base-uncased on prompts labeled with 0 or 1 for both GPT-3.5 and GPT-4. 

I fine-tuned for three epochs with batch size of 16, learning rate of 1e-4, and weight decay of 0.1. I used a V100 GPU on Colab.

After fine-tuning was finished, I optimized the cutoff probability threshold (default 0.5) to maximize the F1 score. The threshold is the limit at which we consider a prompt to be answerable or not by a given model. For example, a probability score of 0.4 means that we do not expect the LLM to be able to answer the prompt correctly. 

Thresholds:

 -  GPT-4: 0.659
 -  GPT-3.5-turbo: 0.604

GPT-4 performance on test set using thresholds:

 - F1 score: 0.889
 -  Accuracy: 0.851
 - Precision: 0.857
 - Recall: 0.924

 GPT-3.5-turbo performance on test set using thresholds:
  - F1 score: 0.674
  - Accuracy: 0.701 
  - Precision: 0.737
  - Recall: 0.620

These results are a lot better than before, but still not great. Regardless, I uploaded the model to Hugging Face.

## Insights

### Readability is useless
Prompt readability does NOT correlate to difficulty! I believe this is because humans are knowledge LOW but reasoning HIGH and LLMs are knowledge HIGH and reasoning LOW. We don't know as much information as an LLM so things like obscure words make questions hard for us to answer. Readability scores generally try to predict things like that. But to an LLM which has seen trillions of tokens and virtually every word in the English language hundreds of thousands of times, the challenge is not words like "discombobulated". They know everything but are relatively bad at reasoning.

Very readable prompts can be extremely challenging for LLMs while very complex-sounding prompts can be extremely easy.
For instance, no LLM out of Phi-2, Mistral-7b, and Mixtral got this right: "A wooden bridge can carry no more than 5000 pounds. A delivery truck filled with identical boxes, each weighing 15 pounds, will pass over the bridge. The combined weight of the driver and the empty truck is 3755 pounds. What is the maximum number of boxes which can be loaded onto the truck while not exceeding the bridge's weight limit?"

None of the words in that prompt are particularly advanced or challenging. Yet LLMs struggle (in this case, mostly with the arithmetic).

On the other hand, it's very easy for LLMs to answer prompts with complex words like "Explain retrosynthetic analysis".

Also, prompt length is pretty poorly correlated with difficulty, but it's something many readability scoring functions pick up on.

### Word-based approaches are useless
Looking back, it was pretty stupid of me to try using TD-IDF. Prompts  are extremely diverse and TF-IDF is just going to pick up on things like numbers and use them as a proxy for difficulty. For instance, most of the challenging questions involved numbers because LLMs struggle with math, so a TF-IDF model would likely be biased to predict prompts containing numbers as hard. While this heuristic might hold true for the training and testing data (which come from a very narrow domain: LLM benchmark prompts), it's a brittle approach. In real-world applications, your users' prompts may cover an extremely wide range of diverse topics and tasks. Many of these user prompts involving numbers could be trivial for LLMs to handle, while plenty of challenging prompts may not contain any numbers at all.
Word-based methods like TF-IDF are inherently limited in their ability to understand the meaning and structure of prompts. I don't think they can distinguish between superficial correlations and the actual factors that determine prompt difficulty. This pushed me to try distilBERT.

## Difficulty is emergent
I truly did not expect anything but I embedded the prompts in my dataset using [Nomic Atlas](https://atlas.nomic.ai/) to see if there was *any* clustering of prompts by difficulty level.

Unsurprisingly not. Embeddings are supposed to separate texts by meaning, and difficulty is unrelated to that. Difficulty seems to be a higher order (not sure if I'm using that right) property of prompts that can't be predicted readily from meaning or words.
![Embedding visualization](https://cloud-bzbbieacu-hack-club-bot.vercel.app/0screenshot_2024-04-26_at_11.29.21___pm.png)
I didn't end up trying this but I wonder if something that might work is asking an LLM to think step by step and plan out the steps to solving a task, and using the number of steps as a proxy for difficulty. But this requires too much time since you need to wait for a large model to output many tokens. I think the general idea behind this is related to the strategy using Find + Replace transformers which I discuss later.

## Try the model
I've uploaded my fine-tuned distilBERT [model](https://huggingface.co/gaodrew/OpenDispatcher_v3_gpt35turbo_and_gpt4) to Hugging Face. Apache 2.0 license! 
And here's a Colab [notebook](https://colab.research.google.com/drive/1UBjMXJn-5X4FPf_dUuTk_dnV66gTg0Em?usp=sharing) to run it with.

## Your turn & a mystery
I have made the Phi-2/Mistral-7b/Mistral dataset available here:
https://drive.google.com/file/d/1AxMAE0BGbHH6NTX6dM50NsRCBzhnWwhi/view?usp=sharing

The RouterBench dataset can be downloaded here: https://huggingface.co/datasets/withmartian/routerbench

As I mentioned at the beginning of this blogpost, the Martian team has a proprietary technique that involves turning Transformers into programs. I don't fully understand what this means or how it works. But it seems to work really well. They share some hints into their methods here: https://blog.withmartian.com/post/mission. Apparently it involves "model mapping" and they are able to train small models to predict the behavior of larger models.

I did a little stalking into the publication history of the founders and I believe their method may have to do with Find + Replace Transformers (see their OpenReview submission [here](https://openreview.net/forum?id=MGWsPGogLH&trk=public_post_comment-text)). I'll add snippets from my conversation with Claude Opus regarding how Find + Replace Transformers could be implemented for model routing. Can you figure out how to implement it?

Thanks for reading! If you thought this was interesting, consider following me on X/Twitter: https://twitter.com/itsandrewgao

And thanks again to [@sershokunin](https://twitter.com/sershokunin?ref_src=twsrc%5Egoogle%7Ctwcamp%5Eserp%7Ctwgr%5Eauthor) for sponsoring! 

If you're interested in sponsoring a project and have one in mind (or just want to support me on a general project), send me a DM on X: https://twitter.com/itsandrewgao.

Relevant Literature:

 - [Large Language Model Routing with Benchmark Datasets](https://arxiv.org/pdf/2309.15789)
 - [RouterBench](https://arxiv.org/html/2403.12031v1)
 - [Introducing RouterBench](https://blog.withmartian.com/post/router-bench)
 - [Tryage](https://arxiv.org/abs/2308.11601)

## Appendix: Find + Replace Transformers

### Explaining the paper
1.    
    Transformers are not Turing complete: The authors prove that transformers like GPT have bounded computational power. Once the architecture is fixed, there is a limit to the computation it can perform. This means transformers will fail to generalize beyond a certain input size and complexity. The fundamental reason is that transformers are stateless - given the same input, they must produce the same output. This constrains their computation.
    
2.  Find+Replace transformers are Turing complete: The authors introduce a new architecture called Find+Replace transformers. These are multi-transformer systems that can provably simulate any Turing machine, making them Turing complete. The basic idea is to have "Find" transformers that identify relevant subsequences in the input, and "Replace" transformers that take those subsequences and output a replacement. By chaining Find and Replace operations, they can implement any computable function.
    
3.  Find+Replace transformers outperform GPT-4: The authors demonstrate that Find+Replace transformers, despite being much smaller, can outperform GPT-4 on several challenging reasoning tasks like Tower of Hanoi, arithmetic, and dynamic programming. The Turing completeness allows them to generalize better to more complex variations of these problems compared to GPT-4.
    
4.  Transformers can be compiled into Find+Replace systems: An interesting consequence is that arbitrary programs can be compiled into Find+Replace transformers. This could aid interpretability research by allowing known programs to be encoded in transformers.
    

In summary, the key insight is that allowing transformers to maintain state and perform open-ended computation via interaction between multiple transformers can make the overall system Turing complete. This expanded computational power manifests in better generalization ability compared to state-of-the-art language models like GPT-4. The Find+Replace transformer is an example architecture that achieves Turing completeness and demonstrates superior performance empirically.

### Focusing on Find + Replace Transformers
The Find+Replace transformer architecture consists of three main components:

1.  Find Transformers: These transformers have a fixed context length and scan the input sequence to identify relevant subsequences. They output a high activation (e.g., all 1s) for the selected subsequence and low activation (e.g., all 0s) for the rest.
    
2.  Replace Transformers: These transformers take the subsequences identified by the Find transformers as input. They output a replacement subsequence of a fixed length, which is used to replace the subsequence identified by the first Find transformer.
    
3.  Map: This is a function that maps each Replace transformer to an ordered set of Find transformers. It determines which Find transformers provide input to each Replace transformer.
    

The Find+Replace transformer operates on a sequence called the "tape," which can be of arbitrary length. The computation proceeds as follows:

1.  The Find transformers scan the tape and identify relevant subsequences.
2.  The Replace transformers take these subsequences as input, as determined by the Map function.
3.  The first Replace transformer outputs a subsequence that replaces the one identified by the first Find transformer in its associated set.
4.  This process continues iteratively until a halting condition is met.

To prove that Find+Replace transformers are Turing complete, the authors show that they can simulate any Tag System, which is a simple computational model known to be Turing complete. They do this by:

1.  Representing the Tag System's alphabet using the Find+Replace transformer's vocabulary.
2.  Using Find transformers to identify the first two symbols and the end of the sequence.
3.  Using Replace transformers to remove the first two symbols and append the production rule corresponding to the first symbol.

By iteratively applying these steps, the Find+Replace transformer can simulate the Tag System, and thus, it is Turing complete.

### A Plan
Here's a detailed plan for using the topics mentioned in the Find+Replace Transformers paper to develop a language model routing system that predicts whether a given transformer model can successfully complete or solve a provided prompt/task:

1.  Collect a diverse dataset of prompts/tasks:
    
    -   Gather a large dataset of prompts and tasks covering various domains, such as question answering, language translation, summarization, arithmetic, and logical reasoning.
    -   Ensure that the dataset includes both tasks that transformer models can solve and tasks that are beyond their capabilities.
    -   Label each prompt/task with the corresponding transformer models that can successfully complete it.
2.  Analyze the computational complexity of the prompts/tasks:
    
    -   Develop a framework to analyze the computational complexity of each prompt/task in the dataset.
    -   Identify key characteristics that contribute to the complexity, such as input size, required number of steps, and the need for maintaining state or performing open-ended computation.
    -   Categorize the prompts/tasks based on their computational complexity (e.g., regular, context-free, context-sensitive, Turing-complete).
3.  Train a Find+Replace transformer model for complexity analysis:
    
    -   Develop a Find+Replace transformer architecture tailored for analyzing the computational complexity of prompts/tasks.
    -   Train the model on the labeled dataset to predict the complexity category of a given prompt/task based on its characteristics.
    -   Evaluate the model's performance and fine-tune it to improve accuracy.
4.  Build a database of transformer model capabilities:
    
    -   Create a database that stores information about the capabilities of various transformer models, such as GPT-3, GPT-4, and the Find+Replace transformer.
    -   For each model, record its architecture, training data, computational power, and performance on different types of tasks.
    -   Continuously update the database as new transformer models are developed and their capabilities are assessed.
5.  Develop a routing algorithm:
    
    -   Create an algorithm that takes a prompt/task as input and predicts which transformer model is most likely to successfully complete or solve it.
    -   Use the Find+Replace transformer model to analyze the computational complexity of the input prompt/task.
    -   Match the complexity category with the capabilities of the transformer models in the database.
    -   Select the most suitable transformer model based on factors such as computational power, task-specific performance, and availability. 

### Why the plan might work
The main goal of the plan is to create a system that can predict which transformer model is most likely to successfully complete a given prompt or task. This is important because different transformer models have different capabilities and are better suited for certain types of tasks.

The plan works by leveraging the key ideas from the Find+Replace Transformers paper:

1.  Computational complexity matters: The paper shows that the computational complexity of a task determines whether a transformer model can generalize to solve it. Tasks that require open-ended computation or maintaining state are more complex and may not be solvable by traditional transformer models, which are not Turing-complete.
    
2.  Find+Replace transformers can analyze complexity: The paper introduces Find+Replace transformers, which are Turing-complete and can analyze the computational complexity of tasks. By training a Find+Replace transformer to predict the complexity category of a prompt/task, we can gain insights into whether a traditional transformer model is likely to solve it.
    
3.  Matching complexity with model capabilities: By creating a database of transformer models and their capabilities, we can match the complexity of a task with the appropriate model. If a task is identified as requiring open-ended computation, we can route it to a Find+Replace transformer or a model with similar capabilities, while simpler tasks can be routed to traditional transformer models.
    

Here's a step-by-step explanation of why the plan works:

1.  Collecting a diverse dataset of prompts/tasks ensures that the system is trained on a wide range of examples and can handle various types of inputs.
    
2.  Analyzing the computational complexity of the prompts/tasks helps in understanding the inherent difficulty and requirements of each task. This information is crucial for matching tasks with the appropriate transformer models.
    
3.  Training a Find+Replace transformer model for complexity analysis allows the system to automatically predict the complexity category of new prompts/tasks. This is based on the idea from the paper that Find+Replace transformers can analyze computational complexity.
    
4.  Building a database of transformer model capabilities provides a way to match the complexity of a task with the appropriate model. By knowing the strengths and limitations of each model, the system can make informed decisions about which model is most likely to succeed.

### How can Find + Replace Transformers predict computational complexity classes?
In the paper, the authors prove that Find+Replace transformers are Turing-complete, meaning they can simulate any Turing machine and perform any computable task. This is in contrast to traditional transformer models, which are not Turing-complete and have limitations in their computational expressiveness.

The key idea is that the computational complexity of a task determines whether a transformer model can generalize to solve it. Tasks that require open-ended computation, maintaining state, or a large number of steps are more complex and may not be solvable by traditional transformer models.

Find+Replace transformers, being Turing-complete, can analyze the computational complexity of tasks by simulating their execution and determining the resources required (e.g., time, memory, or number of steps). This analysis can be used to predict whether a traditional transformer model is likely to solve the task successfully.

Here are a couple of examples to illustrate this concept:

Example 1: Arithmetic Task
Let's consider a task that involves adding two large numbers, each with 100 digits. This task requires a significant number of steps and maintaining state (i.e., carrying over digits) during the computation.

A Find+Replace transformer can simulate the addition process step-by-step, keeping track of the intermediate results and the number of steps required. By analyzing the computational resources needed, it can determine that this task falls into a complexity class that traditional transformer models may struggle with, such as TC0 (threshold circuit) or beyond.

Based on this analysis, the routing system would predict that a traditional transformer model is unlikely to solve this task accurately and would instead route it to a Find+Replace transformer or a model with similar Turing-complete capabilities.

Example 2: Sequence Reversal Task
Consider a task that involves reversing a sequence of characters, where the input sequence can be of any length. This task requires maintaining state (the reversed sequence) and the number of steps grows linearly with the input size.

A Find+Replace transformer can simulate the reversal process and analyze the computational complexity. It can determine that this task belongs to a complexity class like LOGSPACE (logarithmic space) or P (polynomial time), which are solvable by traditional transformer models.

In this case, the routing system would predict that a traditional transformer model is likely to solve this task successfully, and it would route the task accordingly.

It's important to note that the actual implementation of using Find+Replace transformers for complexity analysis would involve training them on a labeled dataset of tasks with known complexity classes. The transformer would learn to predict the complexity class based on the task's characteristics, such as input size, required number of steps, and the need for maintaining state.

Once trained, the Find+Replace transformer can be used to analyze the complexity of new, unseen tasks and provide predictions to the routing system. These predictions, along with the database of transformer model capabilities, enable the routing system to make informed decisions about which model is most likely to succeed for a given task.

### What's a computational complexity class?
A complexity class is a set of problems that can be solved by an abstract machine using a certain amount of a computational resource, such as time or memory. Complexity classes are used to categorize problems based on their inherent difficulty and the resources required to solve them.

Here are some common complexity classes:

1.  P (Polynomial Time): The set of problems that can be solved by a deterministic Turing machine in polynomial time. These are considered "efficiently solvable" problems.
    
2.  NP (Non-deterministic Polynomial Time): The set of problems that can be verified by a non-deterministic Turing machine in polynomial time. These problems are often challenging to solve but easy to verify once a solution is found.
    
3.  LOGSPACE (Logarithmic Space): The set of problems that can be solved by a deterministic Turing machine using a logarithmic amount of memory space.
    
4.  PSPACE (Polynomial Space): The set of problems that can be solved by a deterministic Turing machine using a polynomial amount of memory space.
    
5.  EXP (Exponential Time): The set of problems that can be solved by a deterministic Turing machine in exponential time.
    
6.  BPP (Bounded-error Probabilistic Polynomial Time): The set of problems that can be solved by a probabilistic Turing machine in polynomial time with a bounded probability of error.
    
7.  BQP (Bounded-error Quantum Polynomial Time): The set of problems that can be solved by a quantum computer in polynomial time with a bounded probability of error.
    

These complexity classes form a hierarchy, with some classes being subsets of others. For example, P is believed to be a proper subset of NP, and NP is a subset of PSPACE.

Understanding complexity classes is crucial for determining the inherent difficulty of problems and selecting appropriate algorithms or models to solve them. In the context of the language model routing system, the Find+Replace transformer's ability to analyze the computational complexity of tasks and identify their complexity classes helps in predicting whether a traditional transformer model is likely to solve the task successfully.

###
Some theoretical code cooked up by Claude Opus:
https://pastebin.com/m6X1PBAQ
