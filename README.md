# FAKE NEWS demo


The idea is to recreate an UI for the following experiment and to extend it with : 
- A final page that present the results 
- Possibly a functionality to have a LLM answering the question (later)

https://bias.ulb.be/p/q3245dzo/intro/Example/2

The demo would only cover the first part of the of the experiment that deals with headline concerning : ages, gender and ethnicity. 

All headline can be seen [here](./code/headlines.csv)

After having answered the question, the use would be asked it ethnicity, age and gender. He can choose to not answer. After the we would have a page showing it performance and how it ranks compared to different algorithm (Wich graph should be displayed is still to be decided)


The original experiment is described in [Experimental design](./Experimental_Design.pdf).

AI summary : 

This experimental setup describes a study designed to test how algorithmic approaches can help improve collective decision-making while mitigating individual and group biases. The researchers created an online platform where participants evaluate whether news headlines are real or fake, particularly headlines involving sensitive characteristics like gender, ethnicity, and age. Each participant sees a set of 48 headlines (balanced across different sensitive groups and between real/fake) and rates how likely each headline is to be real on a 5-point scale from "very unlikely" to "very likely." The experiment will recruit 200 paid participants through Prolific. The collected responses will be used to compare different aggregation algorithms (including EXP4, MetaCMAB, and simple averaging) to see which best combines individual judgments into accurate collective decisions. Preliminary results with 85 voluntary participants showed that the MetaCMAB algorithm generally outperformed other approaches. Beyond testing algorithm performance, the study will examine patterns in how different demographic groups respond, whether response times correlate with accuracy, and if diverse groups make better collective decisions than homogeneous ones. This setup allows testing both the technical performance of aggregation algorithms and broader questions about human bias and collective intelligence.


More info can be read in the [paper](./Fake_news_paper.pdf).

AI summary : 

The paper investigates how individual and societal biases affect people's ability to identify true versus false news headlines, particularly those involving sensitive characteristics like gender, age, and ethnicity. The researchers conducted experiments where participants evaluated news headlines, with some headlines being authentic and others modified to swap the sensitive groups mentioned (e.g., changing "men" to "women"). Their analysis revealed that demographic factors, headline categories, and information presentation significantly influenced judgment errors. While the "wisdom of crowds" concept suggests collective diversity might counteract individual biases, they found that prevalent societal beliefs often dominated, leading to collective errors. However, the researchers demonstrated that adaptive machine learning algorithms (specifically MetaCMAB and ExpertiseTree) could help mitigate these biases by dynamically adjusting how different participants' opinions were weighted in collective decision-making. These algorithms not only improved overall accuracy but also reduced the impact of framing effects and group biases, providing a promising approach for enhancing collective decision-making in sensitive contexts.
