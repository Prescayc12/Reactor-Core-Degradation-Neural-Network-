Title: Predicting Reactor Pressure Vessel (RPV) Embrittlement via Multi-Layer Perceptron (MLP)

Introduction: Many nuclear reactors in the United States are currently extending their operational lifecycles to 80 years, far beyond the 40-year spans planned during their construction in the 1970s, 80s, and 90s. A primary life-limiting factor for these plants is the degradation of the RPV. Unlike other components, the RPV is not replaceable, and subject to intense neutron irradiation and high temperatures, which induce material brittleness. Specifically, this manifests as a shift in the Ductile-to-Brittle Transition Temperature (DBTT). Accurately predicting this shift is vital for regulatory confidence and structural integrity, ensuring that vessels remain safe and resilient. 

Problem Definition:
The challenge lies in the complex, non-linear interaction between material composition (like copper and nickel) and environmental conditions (neutron flux). Traditional semi-empirical models often struggle to capture the synergistic effects of solute cluster formation at high fluences. 

The objectives are:
1. Develop a machine learning model capable of capturing non-linear material-irradiation synergies. 
2. To predict the DBTT shift with higher accuracy than current algorithms
3. Provide a data-driven tool for reactor lifetime extension decisions

Data Acquisition and Format: This primary data source is the NRC's Power Reactor Embrittlement Database (PRED), which contains experimental results from surveillance capsules across U.S. reactors. The inputs are chemical weight percentages for Cu, Ni, Mn, Si, irradiation temperature, and neutron flux. The output is a value representing the shift in DBTT. 

Proposed Architecture: I propose an MLP architecture to perform this regression task. The model will consist of an input layer corresponding to the material and environment features, followed by three hidden layers. A Rectified Linear Unit (ReLU) activation function will be used for the hidden layers to model non-linearities. To prevent overfitting to historical data, dropout and L2 regularization will be used to ensure the model generalizes to modern applications. 

Evaluation Plan: Root mean square error and mean absolute errors will quantify the error in DBTT shift, and an R-squared score will assess the variance captured compared to Nuclear Regulatory Commission (NRC) values (Eason et al. 2013). K-fold cross validation will be used to ensure stability across the dataset. Predicted vs measured plots will be analyzed to identify biases. 

Reference: 

Eason, E. D., et al. (2013). A Physically Based Correlation of Irradiation-Induced Transition Temperature Shift for Reactor Pressure Vessel Steals. NRC Report. 
