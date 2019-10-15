NBAPredictor Source Code
------------------------
By: Andras Palfi

This program has the ability to try to predict the outcome of previous NBA Games using either a Deep Neural Network (DNN)
 or a Support Vector Machine (SVM). To date, it the programs best model has been able to achieve a 73% accuracy on
 predicting a subset of games in the 2017-2018 season. It allows users to run experiments in a very modular way, allowing
 the user to change options such as the years of NBA data that should be tested/trained on, which method to use (DNN or SVM),
 the learning rate, and much more. The full list of configurable options are outlined in the projects config file
 (./NBAPredictor/resource/config.ini).

The project has the following dependencies:

    * TensorFlow (either version 1 or 2) - Tasked with running the Neural Network
    * Pandas - Data manipulation
    * Sklearn - Runs the SVM
    * NumPy - Data manipulation

This program is tested to run on macOS and Ubuntu Linux. There is no gurantee of compatibility in Windows.

By far the easiest way to run this program is to use Anaconda 3 (https://www.anaconda.com/distribution/) since it will
install all dependencies for you. If you have conda3 installed, run the following from the `./NBAPredictor` directory:

    For macOS:

    `conda create --name NBAPredictor --file spec-file-macos.txt`

    For Linux:

    `conda create --name NBAPredictor --file spec-file-linux.txt`

    And then:

    `conda activate myenv`

If you do not desire to use conda, ensure your Python interpreter has access to all the dependencies listed above. The
default configuration in NBAPredictor will run a DNN with 3 hidden layers (24-24-10 nodes, respectivley) for a
pre-configured amount of instances. To run the program, execute:

    ```
    cd NBAPredictor
    ./main.py
    ```

    