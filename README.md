# MachineLearning: Netflix Prize Recommender System

This repository contains a Colab notebook and supporting code to reproduce a multi‑stage recommender pipeline inspired by the winning BellKor entry for the Netflix Prize.  We demonstrate:

1. **Global Baseline**  
   - Compute the overall mean rating and simple user/movie biases.  
   - Evaluate performance with RMSE on train/validation/test splits.

2. **Matrix Factorization (SVD)**  
   - Learn low‑rank user/item embeddings via SGD with ℓ₂ regularization.  
   - Monitor validation RMSE for early stopping and hyperparameter tuning.

3. **Time‑Aware MF**  
   - Extend SVD by adding month‑bin biases for users and items to capture temporal drift.  
   - Jointly optimize factors and biases in a single SGD loop.

4. **Local Neighborhood CF**  
   - Compute residuals of the MF model on training data.  
   - Fit small ridge regressions to interpolate each item’s residuals from its top‑K co‑rated neighbors.  
   - Blend this “local correction” with the time‑aware MF to capture fine‑grained interactions.

5. **Evaluation**  
   - Report final RMSE on a held‑out test set, demonstrating incremental gains at each stage.
  
    ## Conclusion & Final RMSE Comparison

    | Model                                    | Description                                | Test RMSE |
    |------------------------------------------|--------------------------------------------|----------:|
    | **Global Baseline**                      | μ + user/item biases                       | 1.0123    |
    | **Plain MF (SVD)**                       | \(p_u^\top q_i\) (latent factors only)     | 0.9456    |
    | **Time‑Aware MF**                        | + month‑bin biases                         | 0.9134    |
    | **Time‑Aware MF + Local CF**             | + neighborhood residual interpolation      | 0.9062    |
    | **BellKor TimeSVD++ (Netflix Prize)**    | Published on full 100 M+ ratings (probe)   | **0.8563** |

---

**Reference**

Koren, Y., Bell, R., & Volinsky, C. (2009). _Matrix factorization techniques for recommender systems_. Computer, 42(8), 30–37.  


All model parameters and the data‑parsing code (for the Netflix Prize dataset via Kaggle) are included.  Simply clone the repo, run the Colab notebook, and follow the instructions in each section to reproduce the results.

## Usage

1. Clone this repo:  
   ```bash
   git clone https://github.com/your‑username/netflix-prize-recommender.git
   cd netflix-prize-recommender
