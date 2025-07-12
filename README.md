# Sleep_Quality_Survey
Analysing data from a survey on sleep quality in relation to what the person did before bed.  For this project, the data will be cleansed first, then sorted into groups of similar responses for analysis. _[View survey-sleep.xlsx](data-and-code/survey-sleep.xlsx)_ from https://caplena.com/en/blog/text-analytics-getting-started-with-datasets

## Modules Used:
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit-learn?style=for-the-badge&logo=scikit-learn&logoColor=white)
[![Sentence Transformers](https://img.shields.io/badge/Sentence_Transformers-HF%20Models?style=for-the-badge)](https://huggingface.co/sentence-transformers)

## The Code 
Importing the libraries used in this project.
```python
import pandas as pd 
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from collections import Counter
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
```

The columns ID, OS, and Country contain statistics that have no substantial influence on respondents' sleep quality.  These columns are removed from the dataframe. The dataframe is then visualised in plotly. 
```python
data = pd.read_excel("survey-sleep.xlsx")
data = data.drop(columns=['ID', 'OS','Country'])
table1 = go.Figure(data=[go.Table(
    header=dict(values=list(data.columns),
                fill_color='paleturquoise',
                align='left'),
    cells=dict(values=[data.Area, data.Gender, data.Age, data['How did you sleep last night?'], data['What did you do yesterday that could have influenced your sleep quality - positively or negatively?'], data['Marital Status'], data['Number of children'], data.Education, data['Employment Status']],
               fill_color='lavender',
               align='left'))
])
pio.renderers.default = "browser"
table1.show()
```

<img src="images/Table1.png" alt="Plot" width="70%"/>



