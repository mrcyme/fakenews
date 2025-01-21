Code supporting "Mitigating Biases in Collective Decision-Making: Enhancing Performance in the Face of Fake News". 

Our results were obtained with ```python3.7```

```analysis.ipynb``` contains code in support of our analysis

```cdmsimulation.ipynb``` implements our simulated cdm setting

Requirements are given in ```requirements.txt``` and can be installed through ```pip install -r requirements.txt```

Participant responses are given in ```responses.csv```, whose columns match the descriptions below

| column | name	description | 
| ------------- | ------------- |
| treatment	| identifier for the set of headlines presented to the participant | 
| trial	| trial/round in which the headline was presented  | 
| arm	| which "arm" the headline was presented as (0=left, 1=middle, 2=right) | 
| advice	| the participant's response (0=very unlikely, 0.25=unlikely, 0.5=undecided, 0.75=likely, 1=very likely) | 
| genuine	| whether the headline was genuine (1) or altered (0) | 
| headline	| the headline as shown to the participant | 
| original	| the headline without before a possible alteration | 
| expert_id	| participant's identifier | 
| sentiment	| whether the headline reported a negative (-1) or positive (1) outcome | 
| expert:ethnicity	| the participant's ethnicity | 
| expert:sex	| the participant's sex | 
| expert:age	| the participant's age | 
| outcome:white, outcome:black, outcome:young, outcome:old, outcome:male, outcome:female	| whether the headline reported a negative (-1) or positive (1) or neutral (0) outcome for the specified group | 
| trial_time	| how long the participant took to respond to the trial/round | 
