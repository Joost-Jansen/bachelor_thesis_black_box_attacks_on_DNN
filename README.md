# An Empirical Look at Gradient-based Black-box Adversarial Attacks on Deep Neural Networks Using One-point Residual Estimates

This repository contains the code and dataset for the Bachelor's thesis project titled "[An Empirical Look at Gradient-based Black-box Adversarial Attacks on Deep Neural Networks Using One-point Residual Estimates](Research_project_Joost_Jansen_final_paper.pdf)" by Joost Jansen, under the supervision of Stefanie Roos, Jiyue Hang, and Chi Hong at the EEMCS, Delft University of Technology, The Netherlands, completed on June 16, 2022.

### Overview
This research investigates the efficiency of one-point residual estimates in gradient-based black-box adversarial attacks against Deep Neural Networks (DNNs), focusing on reducing the number of queries to the attacked model while maintaining attack accuracy. The study evaluates the performance of the proposed method against traditional two-point estimates using the MNIST and Fashion-MNIST datasets.

### Setup
To replicate the experiments, ensure you have the following prerequisites installed:
- Python 3.7 or later
- PyTorch 1.2.0
- torchvision 0.4.0 (Note: Updated from 0.2.2 to support PyTorch 1.2.0)
- NumPy

### Installation
1. Clone this repository.
2. Install the required dependencies using `pip install -r requirements.txt`.

### Running Experiments
Use the `main.py` script to conduct experiments. You can modify the script to switch between datasets, attack methods, and to adjust hyperparameters.

### Data
This project utilizes the MNIST and Fashion-MNIST datasets for evaluating adversarial attack efficacy. Both datasets are publicly available and can be automatically downloaded through torchvision.

### Contributing
If you find any bugs or have suggestions for improvement, please open an issue or submit a pull request.

### License
This project is open-sourced under the MIT License. See the LICENSE file for more details.

### Acknowledgements
Special thanks to my supervisors Stefanie Roos, Jiyue Hang, and Chi Hong for their guidance and support throughout this project. Also, gratitude to the original authors of the method and the community for their valuable resources and tools.

---

For more detailed information, refer to the [original research paper](https://github.com/ChiHong-Xtautau/BRP-AdversarialAttacks).
