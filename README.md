# PMLDL Assignment 2 (Movie Recommender System)

## Author
* **Name:** Albert Khazipov
* **Group:** DS21-01
* **Email:** a.khazipov@innopolis.university

## Dataset
[MovieLens 100K dataset](https://grouplens.org/datasets/movielens/100k/)


## Repository structure
* `benchmark`: contains evaluation data and test loss calculation.
* `data`: contains initial and preprocessed data
* `notebooks`: contains files with data preparation and training process
* `models`: best trained pytorch model
* `references`: contains a refernce to the dataset
* `reports`: contains figures and final report about work done

## Running the benchmark

- Download a repository `https://github.com/1khazipov/movie_recommender_system`
- Install the libraries `pip install -r requirements.txt`
- Go to `benchmark` folder and run `python evaluate.py`. Will be tested on model from `models/best_pytorch_model.pth`