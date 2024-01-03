## Forest firefighting Multiagent Network

### Dependencies
    
    1. torch 
    2. mathplot
    3. Pygame for visualisation 
 

### How to run
    
    1. Import the model file into program in which you want to run , index.py is used for this purpose here
    2. Essaentially you will have to import model from model.py and then you can
        a. Either train it using model.train()
        b. Or test it by passing test parameter and calling test function. You require pygame to visualize the game. 
        c. Pretrained or early trained model can also be trained further by passing filename in MODEL constructor

### Files
    1. model.py -> implemenntation of MADQN model
    2. simulator.py -> for visualizing using pygame 
    3. utlity.py -> for implementing utilities
    4. best_model.tar -> It storesthe best model we have trained so far. It can be loaded for further training and also for testing and visualization 

#### Simulators folder
    1. This is open source directory which has implementation of map and many other simulation helpers. 
    2. Note: This is third party and not build by us.
    3. Link : https://github.com/rhaksar/simulators 
