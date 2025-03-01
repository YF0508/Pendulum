# Pendulum
```bash
conda env create -f environment.yml
```

If some error messages from Anaconda are raised, you could choose to install the required python3 package manually. Run the following command with CMD in Windows or Shell in Linux or MacOS:

```bash
pip3 install pytorch pygame gym opencv_python
```

How to use
Enter the DQN directory, and run the python3 command 'python3 train.py':

```bash
cd DQN-pytorch # 
python3 train.py
```

When testing the bulit environment, you could let the code idle with the following command:

```bash
python3 train.py --idling
