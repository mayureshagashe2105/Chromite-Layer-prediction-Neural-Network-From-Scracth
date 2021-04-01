<font face="Comic sans MS"><h1 style="text-align:center"><b><centre>Chromite Layer Classification</centre></b></h1><p></p>
<p style="font-size:115%;">In this end-to-end project tutorial, we will train an artificial neural network from scratch as a binary classifier for chromite layers prediction, while understanding the mathematics involved that allows the model to make predictions. Data used for the purpose of training the model can be accessed from <a href="https://www.kaggle.com/saurabhshahane/multivariate-geochemical-classification", target="_blank">here</a>.</p></font>
<font face="Comic sans MS"><p><b>NOTE:</b> Only the Following selected features are used for the purpose of training<br> <table padding="30px"><tr>
    <th>Features</th>
    <th>Description</th>
    </tr>
    <tr><td>Motherhole</td><td>Type of the Mother hole observed in image</td></tr>
    <tr><td>Holetype</td><td>Type of the hole observed</td></tr>
    <tr><td>DepthFrom</td><td>Depth from the Earth at which first observed</td></tr>
    <tr><td>DepthTo</td><td>Final depth at which ore is observed</td></tr>
    <tr><td>Cr2O3_%</td><td>Percentage of Cr2O3</td></tr>
    <tr><td>FeO_%</td><td>Percentage of FeO</td></tr>
    <tr><td>SiO2_%</td><td>Percentage of Si02</td></tr>
    <tr><td>MgO_%</td><td>Percentage of MgO</td></tr>
    <tr><td>Al2O3_%</td><td>Percentage of Al2O3</td></tr>
    <tr><td>CaO_%</td><td>Percentage of CaO</td></tr>
    <tr><td>P_%</td><td>Percentage of P</td></tr>
    <tr><td>Au_ICP_ppm</td><td>Inductive Coupled Plasma analysis of Au</td></tr>
    <tr><td>Pt_ICP_ppm</td><td>Inductive Coupled Plasma analysis of Pt</td></tr>
    <tr><td>Pd_ICP_ppm</td><td>Inductive Coupled Plasma analysis of Pd</td></tr>
    <tr><td>Rh_ICP_ppm</td><td>Inductive Coupled Plasma analysis of Rh</td></tr>
    <tr><td>Ir_ICP_ppm</td><td>Inductive Coupled Plasma analysis of Ir</td></tr>
    <tr><td>Ru_ICP_ppm</td><td>Inductive Coupled Plasma analysis of Ru</td></tr>
    <tr><td>Filter</td><td>Filter</td></tr>
    </table></font>
    <h1>Architecture of the Neural Network</h1>
    Model summary:
    <table>
    <tr><th>Layer Number</th><th>Name Of The Layer</th><th>Number Of Nodes</th><th>Trainable Parameters</th><th>Activation Function</th></tr>
    <tr><td>1</td><td>Input Layer</td><td>19</td><td>0</td><td>None</td></tr>
    <tr><td>2</td><td>Hidden Layer 1</td><td>1,024</td><td>20,480</td><td>Sigmoid</td></tr>
    <tr><td>3</td><td>Hidden Layer 2</td><td>512</td><td>524,800</td><td>Sigmoid</td></tr>
    <tr><td>4</td><td>Output Layer</td><td>2</td><td>1,026</td><td>Softmax</td></tr>
    </table>
    <h1>Key Project Takeaways</h1>
       This project provided hands-on experience in real-time data handling and on the following Machine Learning Techniques:
   <ul>Data wrangling & preprocessing</ul>
   <ul>Mathematics behind Backpropagation</ul>
   <ul>Using differnet activation functions in the same model</ul>
   <ul>Building an efficient Binary Classifier model from scratch using NumPy</ul>
