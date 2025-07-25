# Song Popularity Inference Project

This repository contains the final course project for **STA 141A — Fundamentals of Statistical Data Science**. We analyze over 18,000 tracks from a Kaggle Spotify dataset to understand which audio features influence a song's popularity. The analysis is contained in `spotify_popularity_analysis.Rmd` and uses the data in `datasets/song_data.csv`.

The report examines ordinary least squares, a logit transformation, and beta regression. A reduced beta regression model ultimately provided the most interpretable fit with similar predictive power to the full model.

---

### Some notable results:

- **Positive effects:** Danceability and loudness slightly increase song popularity.
- **Negative effects:** Higher instrumentalness, speechiness, audio valence, energy and longer song duration correspond with lower popularity scores.
- **Low multicollinearity:** The highest VIF was about 3, so predictors could be retained.
- **Model fit:** The beta regression achieved a pseudo $R^2$ of roughly 0.012, and 10-fold cross validation showed the reduced model performed as well as the full model.

To reproduce the the full analysis, view the PDF report at `reports/song_popularity_report.pdf` or open the R Markdown file (`spotify_popularity_analysis.Rmd`) in RStudio and knit it to HTML or PDF.