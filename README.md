<p align="center">
  <img src="https://github.com/mateomartinez510/Eluvio/blob/f901c910727bfda37d783d81419929eb9385863d/images/social_media.jpg" alt="social_media_image" width="75%" height="75%"/>
</p>
## Coding Challenge: Option 1 - Data Science/ML

TL:DR
[Data Analysis & Notebook](https://www.nba.com/stats/) <br>
[1st Predictive Modeling Notebook](https://www.nba.com/stats/) (target variable: "over_18") <br>
[2nd Predictive Modeling Notebook](https://www.nba.com/stats/) (target variable: "nsfc", a new feature created to better capture all content "Not Suitable For Children") <br>

In the Data Wrangling/EDA(LINK) notebook I cleaned the dataset, removing null values/columns, extracted additional features from the date and title columns, and performed a cursory analysis on all of the features in the dataset. Below is the first five rows of the dataset after wrangling.

<img src="https://github.com/mateomartinez510/Eluvio/blob/main/images/wrangled_dataframe.png" alt="wrangled_dataframe"/>

After completing EDA and some exploratory modeling, I determined that complexity of applying NLP to the "title" feature required a deep learning framework, and conceded that my initial attempt at applying traditional models with a TF-IDF vectors was unsuccessful. My initial intent was to use a BERT deep learning model to predictive the number of "up_votes" a post received, but after numerous attempts of model tuning, the algorithm would only predict values very close to be mean, and determined in this use case (and in general) that the BERT model is not suited for linear regression. 

Thus I changed my target dependent variable to "over_18" and used a BERT algorithm for the model. Upon deeper analysis of the "over_18", it was clear this variable indicated if the content of a post contained an R/adult rating, generally denoted by the terms "NSFW","NFSL","Graphic", and "Graphic Warning". However we I reviewed the titles of posts without an "over_18" rating, many still contained "Graphic" "Warnings". Thus I created an additional variable: "nsfc" (Not Suitable For Children), that labeled all posts with the terms: "nsfw", "nsfl", "graphic", "kill", "execution" , "decapitation", "rape", and "dismember", which I believe more fully captures all content not suitable for children (this time model were to be put into production, business stakeholders would need to be consulted on final terms to qualify censorship).

I ulitimately trained to predictive models, one predicting "over_18" and another predicting "nsfc". For my final predictions I used a stacked ensemble approach, using the output of the BERT model as an input into a Random Forest model to combine the tabular and unstructured data.


[Advanced NBA Statistics](https://www.nba.com/stats/) 
