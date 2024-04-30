This repository contains implementations of three different models: DeepFM, DLRM, and MultiTask. Each model is designed to handle a specific type of data and solve a specific problem. The models are implemented in PyTorch and trained using the PyTorch Lightning library. For hyperparameter tuning of the DLRM model, we use Ray Tune, a Python library for experiment execution and hyperparameter tuning at any scale.

### Repo structure

1. **DeepFM**: This model is implemented in the `deepfm.py` file. DeepFM is a deep learning-based factorization machine model that is commonly used for click-through rate prediction in recommendation systems. It combines the strengths of factorization machines for exploiting feature interactions and deep neural networks for learning high-order feature interactions. For more information about DeepFM, you can refer to the original paper: [DeepFM: A Factorization-Machine based Neural Network for CTR Prediction](https://arxiv.org/abs/1703.04247).

2. **DLRM**: This model is implemented in the `dlrmnet.py` file. DLRM, or Deep Learning Recommendation Model, is a model designed for personalized recommendation. It was introduced by Facebook in the paper [Deep Learning Recommendation Model for Personalization and Recommendation Systems](https://arxiv.org/abs/1906.00091). The model handles both dense and sparse features, making it suitable for datasets that have a mix of numerical and categorical features. For hyperparameter tuning, we use Ray Tune, which is a flexible, scalable, and user-friendly tool for tuning machine learning models. For more information about Ray Tune, you can refer to the [Ray Tune documentation](https://docs.ray.io/en/latest/tune/index.html).

3. **MultiTask**: This model is implemented in the `multitask/mmoe.py` file. The MultiTask model is a multi-task learning model that uses a shared representation to solve multiple tasks simultaneously. The model uses the Multi-gate Mixture-of-Experts (MMoE) method to learn task-specific representations. For more information about MMoE, you can refer to the original paper: [Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts](https://dl.acm.org/doi/10.1145/3219819.3220007).

4. **BST**: This model is implemented in the `bst.py` file. The behavior sequence transformer is a model designed for handling the sequential nature of user interactions in recommendation systems. It uses a transformer-based architecture to capture the temporal dependencies in the user behavior sequences. For more information about the behavior sequence transformer, you can refer to the original paper: [Behavior Sequence Transformer for E-commerce Recommendation in Alibaba](https://arxiv.org/abs/2006.15578).


### Data 

The repository also contains two datasets:

1. **Criteo Dataset**: This dataset is used with the DLRM model. It is a large-scale dataset for click-through rate prediction, containing both numerical and categorical features. The dataset is loaded and processed in the `criteo_data.py` file.

2. **Census Income Dataset**: This dataset is used with the MultiTask model. It is a binary classification dataset where the task is to predict whether a person makes over 50K a year. The dataset contains both numerical and categorical features. The dataset is loaded and processed in the `census_income.py` file.

3. **MovieLens Dataset**: This dataset is used with the BST model. It is a dataset containing user ratings for movies. The dataset is loaded and processed in the `movie_rating_seq.py` file.

### How to Run

To run the models, you can use the following steps:
```commandline
# To run deepfm model
python deepfm/train.py
```

