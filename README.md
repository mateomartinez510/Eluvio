<p align="center">
  <img src="https://github.com/mateomartinez510/Eluvio/blob/f901c910727bfda37d783d81419929eb9385863d/images/social_media.jpg" alt="social_media_image" width="75%" height="75%"/>
</p>

<h3>Coding Challenge: Option 1 - Data Science/ML</h3>

TL,DR: check out the three notebooks below for the Data Wrangling and two predictive models I trained:<br>
[1. Data Wrangling & EDA Notebook](https://www.nba.com/stats/) <br>
[2. Predictive Modeling Notebook I](https://www.nba.com/stats/) (target variable: "over_18") <br>
[3. Predictive Modeling Notebook II](https://www.nba.com/stats/) (target variable: "nsfc", a new feature created to better capture all content "Not Suitable For Children") <br>

In the Data Wrangling/EDA(LINK) notebook I cleaned the dataset, removing null values/columns, extracted additional features from the date and title columns, and performed a cursory analysis on all of the features in the dataset. Below is the first five rows of the dataset after wrangling.

<img src="https://github.com/mateomartinez510/Eluvio/blob/main/images/wrangled_dataframe.png" alt="wrangled_dataframe"/>

The main focus of this analysis was to create predictive model to classify posts as having graphic content not suitable for children. The explanatory variable for this model is text within the "title" column. Below are a few visualizations from the EDA process.

In this first plot, we can see the number of "over_18" posts by year. we see a general trend showing an increase in "over_18" posts per year, which generally corresponds to an overall increase in annual posts. <br>
<img src="https://github.com/mateomartinez510/Eluvio/blob/main/images/over_18_posts_by_year.png" alt="over_18_posts_by_year" width="50%" height="50%"/>

In this next plot, we can see the distribution of characters counts per post, as a histogram and KDE plot. This right skewed distribution shows that the majority of posts contain less than 100 characters, with a minority containing up to 300 characters. For computational efficiency, the deep learning model trimmed all posts to 150 characters.<br>
<img src="https://github.com/mateomartinez510/Eluvio/blob/main/images/char_count_kde_histogram.png" alt="char_count" width="50%" height="50%"/>
 
In the modeling process, we utilized a Bidirectional Encoder Representations from Transformers model (BERT) that created words embeddings to feed into a Keras Tensorflow Sequential model with a pretrained BERT model containing 12-layers, 768-hidden, 12-heads, and 110M parameters. However, as part of the EDA process, we created a TF-IDF vectorized dataset to analyze the most and least frequent terms. Below are plots of the most frequent terms in the whole dataset, and then filtered by "over_18" posts.<br>
<img src="https://github.com/mateomartinez510/Eluvio/blob/main/images/most_common_words_by_tf_idf_vectors.png" alt="tf_idf_common_words"/>

After completing EDA and some exploratory modeling, I determined that complexity of applying NLP to the "title" feature required a deep learning framework, and conceded that my initial attempt at applying traditional models with a TF-IDF vectors was unsuccessful. My initial intent was to use a BERT deep learning model to predictive the number of "up_votes" a post received, but after numerous attempts of model tuning, the algorithm would only predict values very close to be mean, and determined in this use case (and in general) that the BERT model is not suited for linear regression. 

Thus I changed my target dependent variable to "over_18" and used a BERT algorithm for the model. Upon deeper analysis of the "over_18", it was clear this variable indicated if the content of a post contained an R/adult rating, generally denoted by the terms "NSFW","NFSL","Graphic", and "Graphic Warning". However we I reviewed the titles of posts without an "over_18" rating, many still contained "Graphic" "Warnings". Thus I created an additional variable: "nsfc" (Not Suitable For Children), that labeled all posts with the terms: "nsfw", "nsfl", "graphic", "kill", "execution" , "decapitation", "rape", and "dismember", which I believe more fully captures all content not suitable for children (this time model were to be put into production, business stakeholders would need to be consulted on final terms to qualify censorship).

I ulitimately trained to predictive models, one predicting "over_18" and another predicting "nsfc". For my final predictions I used a stacked ensemble approach, using the output of the BERT model as an input into a Random Forest model to combine the tabular and unstructured data.

