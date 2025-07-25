# Drake Music Career Analysis

This repository contains the final project for **STA 141B — Data & Web Technologies for Data Analysis**. It explores how Drake has sustained chart‑topping success for more than a decade by combining audio‑feature data from Spotify with web‑scraped and API‑sourced context from YouTube, Reddit, and music‑chart sites.

## Project goals
- Track Drake’s popularity trajectory since 2010 and visualise major inflection points.
- Cluster his songs by sonic attributes to see whether distinct “styles” emerge.
- Test whether particular audio features are associated with the most‑streamed tracks.
- Gauge audience sentiment toward “old” vs “new” Drake using YouTube and Reddit comments.

## Data sources
Spotify’s public Web API provides track‑level audio features and popularity scores; additional context comes from ChartMasters, YouTube and Reddit APIs, plus secondary references for methods tutorials.

## Methods
- **Exploratory visualisation** of the popularity distribution showed strong left‑skew with a mode around 60 and many tracks above 80.
- **K‑means clustering** (k = 4) on standardised audio features grouped songs into four coherent styles; the elbow plot and KneeLocator confirmed k = 4.
- **Sentiment analysis** of top YouTube comments used NLTK VADER to obtain compound scores for each video.

## Key findings
- *Cluster 0* songs are the most danceable on average, while *Cluster 3* is characterised by higher energy, valence, and speechiness.
- Drake’s catalogue is heavily weighted toward high‑popularity tracks, confirming his sustained mainstream appeal.
- Danceability, energy, and valence are the strongest sonic predictors of streaming success.
- Audience sentiment remains predominantly positive even in videos that critique his career.

