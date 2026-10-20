# Amazon Prime Video: Content Analysis & Movie vs TV Show Classification

An end-to-end exploratory data analysis (EDA) of the Amazon Prime Video catalogue, covering data cleaning, univariate and bivariate analysis, time-based trends, cast/crew insights, title keyword analysis, and a classification model that predicts whether a title is a Movie or a TV Show.

Table of Contents
Project Overview
Dataset
Project Workflow
Key Findings
Predictive Modelling
Tech Stack
How to Run
Limitations & Next Steps
Project Overview

The goal of this project is to understand what kind of content Amazon Prime offers — its mix of movies and shows, genres, ratings, countries of origin, release trends and most frequent contributors — and to test whether a title's metadata alone can predict its content type.

Questions explored:

How is the catalogue split between movies and TV shows?
Which genres, age ratings and production countries dominate?
How has content volume and rating mix changed over time?
Who are the most frequently credited directors and actors?
What do typical movie runtimes and show lengths look like, and what are the outliers?
Can metadata reliably classify a title as a Movie or a TV Show?
Dataset
File	Rows	Columns	Description
titles.csv	9,871	15	One row per title: type, description, release year, age certification, runtime, genres, production countries, seasons, IMDb & TMDB scores/votes/popularity
credits.csv	124,235	5	Cast and crew per title, including person name and role (ACTOR / DIRECTOR)

Notable data quality issues found:

Column	Missing %	Handling
seasons	86.25%	Left as NaN — structurally not applicable to movies
age_certification	65.72%	Filled with explicit Not Rated category (not the mode)
tmdb_score	21.09%	Left as NaN, dropped per analysis
imdb_score / imdb_votes	~10.4%	Left as NaN, dropped per analysis
description	1.21%	Filled with placeholder text
Project Workflow
1. Data Understanding & Cleaning
Inspected shape, dtypes, summary statistics and missing values
Removed 3 exact duplicate rows (9,871 → 9,868 titles)
Parsed stringified lists (genres, production_countries) into real Python lists using ast.literal_eval
Standardised text: trimmed titles, upper-cased type and country codes, title-cased genres
Handled missing values deliberately (no imputation of ratings/scores to avoid biasing distributions)
Derived a release_date column from release_year
2. Univariate Analysis
Movies vs TV Shows split
Titles released per year
Top 10 genres (multi-genre titles exploded)
Age certification frequency and share (bar + donut chart)
3. Bivariate Analysis
Age rating by content type
Top genres by content type, and which genres lean toward TV shows
Top 10 production countries, overall and split by type
Genre × 5-year period heatmap (1990 onwards)
4. Time-Based Trends
Movie and TV show releases over time (1980 onwards)
Shift in rating mix (Kids/Family, Teen, Mature) across 5-year periods
5. Advanced Grouping & Insights
Most credited directors and actors (from credits.csv)
Movie runtime and TV show season distributions, with outlier checks and box plots
Word cloud of title keywords and genre breakdown for titles containing Love, War and Christmas
6. Predictive Modelling
Binary classification: Movie (0) vs TV Show (1) using Logistic Regression and Random Forest
Key Findings

Catalogue composition

The catalogue is heavily movie-driven: 8,511 movies (86.2%) vs 1,357 TV shows (13.8%).
Content spans 1912 to 2022, with 2021 as the peak release year (856 titles).

Genres

Drama (4,762) and Comedy (2,987) lead, followed by Thriller, Action and Romance.
Family (29.6% shows) and Documentation (21.9% shows) are the most TV-show-heavy genres; Horror (4.7%) and Thriller (6.3%) are almost entirely movies.

Ratings

65.7% of titles have no age certification — the single largest category.
Among rated titles, R (1,249) is the most common, followed by PG-13 and PG.

Countries

The US dominates with 5,331 titles, followed by India (1,072), the UK (928) and Canada (530).

Cast & crew

The most credited names (e.g. director Joseph Kane, actors George 'Gabby' Hayes and Roy Rogers) are from classic-era westerns — showing a large back-catalogue of older films.

Duration

Median movie runtime is 91 minutes; most fall between 80 and 105 minutes.
Most shows have just 1 season (median 1, mean 2.8), with long-running outliers such as Great Performances (51 seasons) and NOVA (49).
Runtime outliers include very short entries (1–6 minutes) and very long ones (up to 549 minutes), worth validating before any runtime-based analysis.

Title keywords

Titles containing Love skew Drama/Romance, Christmas skews Comedy/Family, and War spans Drama, Action, Comedy and Documentation rather than only the War genre.
Predictive Modelling

Target: type (Movie = 0, TV Show = 1) Features: release year, runtime, number of genres, number of countries, primary genre, primary country, age certification, IMDb score, TMDB popularity Data: 8,519 rows after dropping missing feature values (7,433 movies / 1,086 shows), 80/20 stratified split, class_weight='balanced' to handle imbalance

Model	Accuracy	TV Show Precision	TV Show Recall	TV Show F1
Logistic Regression	0.970	0.82	0.98	0.89
Random Forest	0.988	0.93	0.98	0.95

Feature importance (Random Forest): runtime dominates (0.62), followed by age_certification (0.17) and imdb_score (0.08). Shows list per-episode runtime, which is naturally much shorter than a film's runtime.

Tech Stack
Python 3
pandas, NumPy — data cleaning and manipulation
Matplotlib, Seaborn — visualisation
WordCloud — title keyword analysis
scikit-learn — preprocessing, Logistic Regression, Random Forest, evaluation
How to Run
Clone the repository
bash
   git clone <your-repo-url>
   cd <your-repo-folder>
Install dependencies
bash
   pip install pandas numpy matplotlib seaborn wordcloud scikit-learn jupyter
Add the data — place titles.csv and credits.csv in the project root (same folder as the notebook).
Launch the notebook
bash
   jupyter notebook

Open the notebook and run all cells top to bottom.

Project Structure
├── titles.csv          # Title-level metadata
├── credits.csv         # Cast & crew credits
├── <notebook>.ipynb    # Full analysis notebook
└── README.md
Limitations & Next Steps
Possible feature leakage in the model: TV-specific ratings (TV-MA, TV-14, etc.) and per-episode runtimes almost directly reveal the content type, which partly explains the very high accuracy. A stricter test would drop or regroup these features and compare results.
Label encoding of genre, country and rating imposes an artificial order; one-hot or target encoding would be more appropriate, especially for Logistic Regression.
Only the primary genre and country are used as model features; multi-hot encoding of all genres could add signal.
Two-thirds of titles are unrated, so rating-based conclusions apply only to the rated subset.
Next steps: cross-validation and hyperparameter tuning, predicting IMDb score (regression), joining credits.csv to analyse whether specific directors/actors are linked to higher ratings, and an interactive Power BI / Tableau dashboard.
