# monte_carlo-cpu-gpu-dl

me trying to do the monty ~~python~~ carlo simulation.  <br>
this notebook loosely follows the flow of   <br>
[This nvidia blogpost](https://developer.nvidia.com/blog/accelerating-python-for-exotic-option-pricing/) <br>
[and its accompanying notebooks.](https://github.com/NVIDIA/fsi-samples/tree/main/gQuant/plugins/gquant_plugin/notebooks/asian_barrier_option)   <br>
but i made a version for snowball options instead. <br>
i also liked using [this lil thingy](https://github.com/t4fita/Barrier-option-pricing/blob/main/main.py) to get a better grasp of what was happening since i do not have a masters in stonk trading formulas :pensive:  <br>
As the name suggests, I not only made a model, but also did some cpu, gpu acceleration before throwing into a deep learning model to let it predict things as well. <br> 
Read order is mc_snow -> dl_snow. First one does CPU & GPU, second builds upon GPU to made dataset to train deep learning model.<br> <br>
![Alt](pics/3o06s4.jpg) <br>
me when i get thrown into a meat grinder <br> <br>