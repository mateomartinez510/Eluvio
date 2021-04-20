<p align="center">
  <img src="https://github.com/mateomartinez510/Eluvio/blob/f901c910727bfda37d783d81419929eb9385863d/images/social_media.jpg" alt="social_media_image" width="75%" height="75%"/>
</p>

<h3>Coding Challenge: Option 1 - Data Science/ML</h3>

TL,DR: check out the three notebooks below for the Data Wrangling and two predictive models I trained:<br>
[Data Wrangling & EDA Notebook](https://github.com/mateomartinez510/Eluvio/blob/main/Eluvio_Challenge_Data_Wrangling_EDA.ipynb) <br>
[Predictive Model I: "over_18" Target Variable](https://github.com/mateomartinez510/Eluvio/blob/main/Eluvio_Challenge_Modeling(BERT_classification_over_18).ipynb) <br>
[Predictive Model II: "nsfc" Target Variable](https://github.com/mateomartinez510/Eluvio/blob/main/Eluvio_Challenge_Modeling(BERT_classification_over_18).ipynb) <br>
("nsfc" is a new feature created to better capture all content "Not Suitable For Children") <br>

### Data Wrangling & EDA(LINK)
In thisnotebook I cleaned the dataset, removing null values/columns, extracted additional features from the date and title columns, and performed a cursory analysis on all of the features in the dataset. Below is the first five rows of the dataset after wrangling.

<img src="https://github.com/mateomartinez510/Eluvio/blob/main/images/wrangled_dataframe.png" alt="wrangled_dataframe"/>

The main focus of this analysis was to create predictive model to classify posts as having graphic content not suitable for children. The explanatory variable for this model is text within the "title" column. Below are a few visualizations from the EDA process.

In this first plot, we can see the number of "over_18" posts by year. we see a general trend showing an increase in "over_18" posts per year, which generally corresponds to an overall increase in annual posts. <br>
<img src="https://github.com/mateomartinez510/Eluvio/blob/main/images/over_18_posts_by_year.png" alt="over_18_posts_by_year" width="50%" height="50%"/>

In this next plot, we can see the distribution of characters counts per post, as a histogram and KDE plot. This right skewed distribution shows that the majority of posts contain less than 100 characters, with a minority containing up to 300 characters. For computational efficiency, the deep learning model trimmed all posts to 150 characters.<br>
<img src="https://github.com/mateomartinez510/Eluvio/blob/main/images/char_count_kde_histogram.png" alt="char_count" width="50%" height="50%"/>

Below we can see a heatmap of the correlation matrix of numeric features in our dataset. Aside from the word count and character count and the Flesch readability score and the Flesch grade level, almost none of the features have much correlation with each other. This would explain the lack of success I had earlier attempting to use TF-IDF vectors with non-deep learning models and the need for the BERT model used in this iteration of this projet.
<img src="https://github.com/mateomartinez510/Eluvio/blob/main/images/heatmap.png" alt="heatmap" width="50%" height="50%"/>

In the modeling process, utilized a Bidirectional Encoder Representations from Transformers model (BERT) that created words embeddings to feed into a Keras Tensorflow Sequential model with a pretrained BERT model containing 12-layers, 768-hidden, 12-heads, and 110M parameters. However, as part of the EDA process, we created a TF-IDF vectorized dataset to analyze the most and least frequent terms. Below are plots of the most frequent terms in the whole dataset, and then filtered by "over_18" posts.<br>
<img src="https://github.com/mateomartinez510/Eluvio/blob/main/images/most_common_words_by_tf_idf_vectors.png" alt="tf_idf_common_words"/>

 ### [Predictive Model I: "over_18" Target Variable](https://www.nba.com/stats/) <br>
Target variable: "over_18" feature <br>

After completing EDA and some exploratory modeling, I determined that complexity of applying NLP to the "title" feature required a deep learning framework, and conceded that my initial attempt at applying traditional models with a TF-IDF vectors was unsuccessful. My initial intent was to use a BERT deep learning model to predictive the number of "up_votes" a post received, but after numerous attempts of model tuning, the algorithm would only predict values very close to be mean, and determined in this use case (and in general) that the BERT model is not suited for linear regression. Thus I changed my target dependent variable to "over_18" and used a BERT algorithm for the model. 

Below is the model summary for the BERT model, which was used for both classifiers. After numerous iterations of model turning, what yielded the best results (optimizing for recall, given the lack of samples from the minority class) was a higher dropout rate, smaller proportion of samples from the majority class, and longer "max_len" for the BERT encoded word embeddings (attempts to use less than 150 characters yielded poor results, a longer length would be beneficial, but ultimately computationally too expensive for this analysis).<br>
<img src="https://github.com/mateomartinez510/Eluvio/blob/main/images/bert_model_summary.png" alt="bert_model_summary" width="50%" height="50%"/>

Below is the loss and accuracy for the training and validation sets during the BERT training process. Given the class imbalances, class weights were used with a 30:1 ratio to improve recall on the minority class.<br>
<img src="https://github.com/mateomartinez510/Eluvio/blob/main/images/over_18_loss_accuracy_training.png" alt="bert_training_loss_acc_over_18" width="50%" height="50%"/>

After training the BERT model, I applied a stacked ensemble model approach to take the outputs of the BERT model and combine them with the other tabular data features from the wrangling process to train a XGBoost model. This stacked model increased precision by 8%. Below is the classification report and confusion matrix from the stacked model.<br>
<img src="https://github.com/mateomartinez510/Eluvio/blob/main/images/over_18_matrix_stacked_model.png" alt="stacked_matrix_over_18" width="50%" height="50%"/>

### [Predictive Model II: "nsfc" Target Variable](https://www.nba.com/stats/) <br>

Upon deeper analysis of the "over_18", it was clear this variable indicated if the content of a post contained an R/adult rating, generally denoted by the terms "NSFW","NFSL","Graphic", and "Graphic Warning". However we I reviewed the titles of posts without an "over_18" rating, many still contained "Graphic" "Warnings". Thus I created an additional variable: "nsfc" (Not Suitable For Children), that labeled all posts with the terms: "nsfw", "nsfl", "graphic", "kill", "execution" , "decapitation", "rape", and "dismember", which I believe more fully captures all content not suitable for children (this time model were to be put into production, business stakeholders would need to be consulted on final terms to qualify censorship).

Below are the loss and accuracy metrics from the training process of the BERT model. This BERT model achieved over 97% accuracy, precision, recall, and F1-score for all metrics. Using the stacked model approach, I was able to increase all metrics to over 98%, which could be a signicicant difference depending on the business application. <br>
<img src="https://github.com/mateomartinez510/Eluvio/blob/main/images/nsfc_loss.png" alt="nsfc_loss_accuracy_metrics" width="50%" height="50%"/>

Below is the classification report and confusion matrix from the stacked model.<br>
<img src="https://github.com/mateomartinez510/Eluvio/blob/main/images/nsfc_classifcation_matrix.png" alt="nsfc_classifcation_matrix" width="50%" height="50%"/>
<br><br>
Here is a [link to the Google Colab](https://drive.google.com/drive/folders/1oQ0-C-A0TbVgPQU6PEDKtv4tnFeM6FJ_?usp=sharing) folder and notebooks used to created this project. <br>
Thank you for taking the time to review my report and analysis. <br>

Kind regards,<br>
Mateo Martinez




