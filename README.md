# X-VLA
Soft-Prompted Transformer as Scalable Cross-Embodiment Vision-Language-Action Model


## Results on Simulations
We evluate X-VLA across 6 simulations, which encompass hundreds of evaluation setups, spanning single-arm, bi-manual robotic systems, autonomous driving, and assessing diverse axes of generalization, including cross-embodiment, cross-environment, and cross-task adaptation.

|Simpler|||Libero|||||Calvin|RoboTwin_2.0||VLABench|NAVSIM|
|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| VM | VM | WidowX |Spatial|Object|Goal|Long|Avg|ABC&rightarrow;D|Easy|Hard|Avg. PS|PDMS|
| 80.4 | 75.7 | 95.8 | 98.2 | 98.6 | 97.8 | 97.6 | 98.1 | 4.43 | 70.0 | 39.0 | 51.1 | 87.3 |

More detailed metrics for each benchmark are available in the following figures. Click to view: [Robotics Simulation](https://github.com/2toinf/X-VLA/blob/main/images/robo.jpeg) and [Autonomous Driving](https://github.com/2toinf/X-VLA/blob/main/images/ad.jpeg). 
