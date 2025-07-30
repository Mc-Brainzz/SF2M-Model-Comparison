# SF2M-Model-Comparison
# A Comparative Analysis of Generative Models for ODE Correction

## 📝 Project Overview
This project compares three different generative models for a common scientific computing task: correcting the output of a biased, low-quality ODE simulation to match a high-fidelity ground truth. We use a 2D "Biased Van der Pol Oscillator" as a benchmark problem.

## 🚀 Models Compared
1.  **SDEdit (from DCSR paper):** A diffusion-based model that learns the structure of the clean data and uses it to denoise a corrupted input.
2.  **DDIB (from Unpaired SR paper):** A diffusion bridge model that learns both distributions and maps between them via a shared latent space.
3.  **[SF]²M (Simulation-Free Score/Flow Matching):** A newer method that learns a direct flow between the two distributions, guided by Optimal Transport.

## 📊 Key Findings
The experiment demonstrates that the **[SF]²M model** provides the best combination of performance and efficiency for this task. It achieved the lowest Wasserstein-2 distance while maintaining a training time comparable to the much less accurate SDEdit model. The DDIB model, while theoretically powerful, proved unstable and inefficient for this chaotic source distribution.

*(Here you can paste a screenshot of your final comparison bar chart)*

## ⚙️ How to Run
1.  Clone the repository: `git clone ...`
2.  Open the main notebook located in the `/notebooks` directory in an environment like Google Colab.
3.  Run the cells from top to bottom to reproduce the experiment and results.
